"""API interne ADLP.
"""
from contextlib import asynccontextmanager
import hmac
import logging
from pathlib import Path
import tempfile
from typing import Annotated, Literal, Optional
import uuid

from fastapi import Depends, FastAPI, File, Form, Header, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer
from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator, model_validator
from starlette.concurrency import run_in_threadpool

from product_rag import MAX_FILE_BYTES, ProductError, identifier
from rag_service import RagService, ServiceError, load_settings

LOG = logging.getLogger('adlp.rag')
MAX_BODY_BYTES = MAX_FILE_BYTES + 64 * 1024


class Query(BaseModel):
    model_config = ConfigDict(extra='forbid')
    question: str = Field(min_length=1, max_length=2000)
    scope: Literal['product', 'common', 'both'] = 'both'
    product_id: Optional[str] = None
    document_id: Optional[str] = None
    doc_id: Optional[Literal['meta', 'arpp', 'methods']] = None
    strategy: Literal['sections'] = 'sections'
    top_k: Annotated[StrictInt, Field(ge=1, le=8)] = 8
    candidate_k: Annotated[StrictInt, Field(ge=1, le=100)] = 20

    @field_validator('question')
    @classmethod
    def nonempty_question(cls, value):
        if not value.strip():
            raise ValueError('Question vide.')
        return value.strip()

    @field_validator('product_id', 'document_id')
    @classmethod
    def valid_ids(cls, value):
        return identifier(value, 'Identifiant') if value is not None else value

    @model_validator(mode='after')
    def valid_context(self):
        if self.scope != 'common' and self.product_id is None:
            raise ValueError('product_id est obligatoire pour product et both.')
        if self.document_id is not None and self.product_id is None:
            raise ValueError('document_id nécessite product_id.')
        if self.doc_id is not None and self.scope != 'common':
            raise ValueError('doc_id est réservé au périmètre common.')
        if self.candidate_k < self.top_k:
            raise ValueError('candidate_k doit être supérieur ou égal à top_k.')
        return self


class DocumentResponse(BaseModel):
    model_config = ConfigDict(extra='allow')
    request_id: Optional[str] = None
    status: Literal['ready']
    user_id: str
    product_id: str
    document_id: str
    source: str
    title: str
    pages: int
    chunks: int
    source_sha256: str
    synthetic: bool


class ProductsResponse(BaseModel):
    request_id: str
    products: list[DocumentResponse]


class SourceResponse(BaseModel):
    source_id: str
    chunk_id: str
    doc_id: str
    document_id: Optional[str] = None
    product_id: Optional[str] = None
    doc_title: Optional[str] = None
    source: str
    source_sha256: Optional[str] = None
    page_start: int
    page_end: int
    doc_type: Optional[str] = None
    scope: Optional[str] = None
    synthetic: Optional[bool] = None
    section: Optional[str] = None
    text: str
    origin: Literal['product', 'common']
    retrieval_score: float
    rerank_score: Optional[float] = None
    retrieval_rank: Optional[int] = None


class RetrievalResponse(BaseModel):
    request_id: str
    question: str
    user_id: str
    product_id: Optional[str] = None
    document_id: Optional[str] = None
    scope: Literal['product', 'common', 'both']
    strategy: Literal['sections']
    sources: list[SourceResponse]
    retrieval: dict


class CitationResponse(BaseModel):
    source_id: str
    quote: str


class AnswerResponse(RetrievalResponse):
    status: Literal['answered', 'partial', 'insufficient']
    answer: str
    citations: list[CitationResponse]
    limitations: list[str]
    validation_passed: bool
    trace_id: str
    model: str
    prompt_version: str
    usage: Optional[dict] = None
    generation_ms: float


class ErrorDetail(BaseModel):
    code: str
    message: str
    request_id: str
    trace_id: Optional[str] = None
    fields: Optional[list[dict]] = None


class ErrorResponse(BaseModel):
    detail: ErrorDetail


class BoundaryMiddleware:
    """Authentifier avant lecture du corps ; borner les uploads, même chunked.

    Au plus 20 Mio + enveloppe multipart en mémoire par requête. Pas de secret
    partagé côté navigateur. /docs décrit la vraie API interne.
    """
    def __init__(self, app, token, body_limit=MAX_BODY_BYTES):
        self.app, self.token, self.body_limit = app, token.encode(), body_limit

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http':
            return await self.app(scope, receive, send)
        request_id = uuid.uuid4().hex
        state = scope.setdefault('state', {})
        state['request_id'] = request_id
        protected = scope.get('path') not in ('/health', '/docs', '/docs/oauth2-redirect', '/openapi.json')
        async def fail(status, code, message):
            response = JSONResponse({'detail': {'code': code, 'message': message, 'request_id': request_id}},
                                    status_code=status, headers={'X-Request-ID': request_id})
            await response(scope, receive, send)
        headers = {}
        for key, value in scope.get('headers', []):
            headers.setdefault(key.lower(), []).append(value)
        if protected:
            auth = headers.get(b'authorization', [])
            if len(auth) != 1 or not auth[0].lower().startswith(b'bearer '):
                return await fail(401, 'unauthorized', 'Jeton de service requis.')
            supplied = auth[0][7:]
            if not hmac.compare_digest(supplied, self.token):
                return await fail(401, 'unauthorized', 'Jeton de service invalide.')
            owners = headers.get(b'x-user-id', [])
            try:
                if len(owners) != 1:
                    raise ValueError('Identité absente ou ambiguë.')
                user_id = identifier(owners[0].decode('ascii'), 'X-User-ID')
            except (ValueError, UnicodeError):
                return await fail(422, 'invalid_user_context', 'X-User-ID doit contenir l’identité vérifiée par le backend.')
            state['user_id'] = user_id
        lengths = headers.get(b'content-length', [])
        if lengths:
            try:
                if len(lengths) != 1 or int(lengths[0]) < 0:
                    raise ValueError()
                if int(lengths[0]) > self.body_limit:
                    return await fail(413, 'request_too_large', 'La requête dépasse la limite autorisée.')
            except ValueError:
                return await fail(400, 'invalid_content_length', 'Taille de requête invalide.')
        body, size = [], 0
        while True:
            message = await receive()
            if message['type'] == 'http.disconnect':
                return
            part = message.get('body', b'')
            size += len(part)
            if size > self.body_limit:
                return await fail(413, 'request_too_large', 'La requête dépasse la limite autorisée.')
            body.append(part)
            if not message.get('more_body', False):
                break
        buffered, delivered = b''.join(body), False
        async def replay():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {'type': 'http.request', 'body': buffered, 'more_body': False}
            return await receive()
        async def with_request_id(message):
            if message['type'] == 'http.response.start':
                message['headers'] = [*(h for h in message.get('headers', []) if h[0].lower() != b'x-request-id'),
                                      (b'x-request-id', request_id.encode())]
            await send(message)
        await self.app(scope, replay, with_request_id)


def error_response(request, status, code, message, trace_id=None, fields=None):
    detail = {'code': code, 'message': message, 'request_id': request.state.request_id}
    if trace_id:
        detail['trace_id'] = trace_id
    if fields:
        detail['fields'] = fields
    headers = {'X-Request-ID': request.state.request_id}
    if status == 503:
        headers['Retry-After'] = '3'
    return JSONResponse({'detail': detail}, status_code=status, headers=headers)


def create_app(settings=None, service=None):
    settings = settings or load_settings()
    service = service or RagService(settings)
    @asynccontextmanager
    async def lifespan(app):
        await run_in_threadpool(service.start)
        try:
            yield
        finally:
            await run_in_threadpool(service.close)
    app = FastAPI(title='ADLP Advanced RAG API', version='1.0.0', lifespan=lifespan, redoc_url=None,
                  description='API interne : Bearer de service + X-User-ID vérifié par le backend web.',
                  responses={code: {'model': ErrorResponse} for code in (401, 404, 409, 413, 415, 422, 429, 500, 502, 503)})
    app.state.rag_service = service
    app.add_middleware(BoundaryMiddleware, token=settings.service_token)
    bearer = HTTPBearer(auto_error=False)
    def context(request: Request, x_user_id: Annotated[str, Header(alias='X-User-ID')],
                credentials=Depends(bearer)):
        return request.state.user_id

    @app.exception_handler(ServiceError)
    async def service_error(request, error):
        return error_response(request, error.status, error.code, error.message, error.trace_id)

    @app.exception_handler(ProductError)
    async def product_error(request, error):
        statuses = {'product_not_ready': 404, 'document_changed': 409,
                    'file_too_large': 413, 'unsupported_file': 415, 'store_busy': 503,
                    'scope_violation': 500, 'incomplete_store': 503,
                    'incompatible_store': 503, 'invalid_catalog': 503}
        return error_response(request, statuses.get(error.code, 422), error.code, str(error))

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, error):
        # Ni corps de requête ni secrets dans le retour d'erreur.
        fields = [{'field': '.'.join(map(str, e['loc'])), 'message': str(e['msg'])[:200]}
                  for e in error.errors()[:10]]
        return error_response(request, 422, 'invalid_request', 'Requête invalide ; consulter le contrat /docs.', fields=fields)

    @app.exception_handler(Exception)
    async def internal_error(request, error):
        LOG.error('RAG request %s failed (%s)', request.state.request_id, type(error).__name__)
        return error_response(request, 500, 'internal_error', 'Erreur interne du service RAG.')

    @app.get('/health', tags=['Service'])
    def health():
        return service.health()

    @app.get('/products', tags=['Documents'], response_model=ProductsResponse)
    def products(request: Request, user_id=Depends(context)):
        return {**service.list_products(user_id), 'request_id': request.state.request_id}

    @app.get('/products/{product_id}/documents/current', tags=['Documents'], response_model=DocumentResponse)
    def current(product_id: str, request: Request, user_id=Depends(context)):
        identifier(product_id, 'product_id')
        return {**service.current(user_id, product_id), 'request_id': request.state.request_id}

    @app.post('/products/{product_id}/documents', tags=['Documents'], response_model=DocumentResponse)
    async def ingest(product_id: str, request: Request, file: UploadFile = File(...),
                     title: Optional[str] = Form(None), synthetic: bool = Form(False),
                     user_id=Depends(context)):
        identifier(product_id, 'product_id')
        try:
            form = await request.form()
            if set(form) - {'file', 'title', 'synthetic'} or any(len(form.getlist(k)) != 1 for k in form):
                raise ServiceError(422, 'invalid_form', 'Champs autorisés : file, title, synthetic, une seule fois chacun.')
            name = (file.filename or '').replace('\\', '/').rsplit('/', 1)[-1]
            if not name or len(name) > 180 or Path(name).suffix.lower() != '.pdf' or any(ord(c) < 32 or ord(c) == 127 for c in name):
                raise ServiceError(415, 'unsupported_file', 'Un fichier PDF est requis.')
            if file.size is not None and file.size > MAX_FILE_BYTES:
                raise ServiceError(413, 'file_too_large', 'Le PDF dépasse 20 Mio.')
            def process_upload():
                with tempfile.TemporaryDirectory(prefix='adlp_upload_') as directory:
                    path = Path(directory) / name
                    file.file.seek(0)
                    total = 0
                    with path.open('wb') as out:
                        while part := file.file.read(64 * 1024):
                            total += len(part)
                            if total > MAX_FILE_BYTES:
                                raise ServiceError(413, 'file_too_large', 'Le PDF dépasse 20 Mio.')
                            out.write(part)
                    return service.ingest(path, user_id, product_id, title=title, synthetic=synthetic)
            result = await run_in_threadpool(process_upload)
            return {**result, 'request_id': request.state.request_id}
        finally:
            await file.close()

    @app.post('/retrieve', tags=['RAG'], response_model=RetrievalResponse)
    def retrieve(query: Query, request: Request, user_id=Depends(context)):
        try:
            return {**service.retrieve(query, user_id), 'request_id': request.state.request_id}
        except ValueError as error:
            if isinstance(error, ProductError):
                raise
            raise ServiceError(422, 'invalid_model_input', 'Entrée trop longue ou incompatible avec les modèles.') from error

    @app.post('/answer', tags=['RAG'], response_model=AnswerResponse)
    def answer(query: Query, request: Request, user_id=Depends(context)):
        try:
            return {**service.answer(query, user_id, request.state.request_id), 'request_id': request.state.request_id}
        except ValueError as error:
            if isinstance(error, ProductError):
                raise
            raise ServiceError(422, 'invalid_model_input', 'Entrée trop longue ou incompatible avec les modèles.') from error
    return app
