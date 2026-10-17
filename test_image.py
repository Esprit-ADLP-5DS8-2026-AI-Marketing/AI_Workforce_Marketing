import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("ERROR: GEMINI_API_KEY is missing")
    exit()

print("API key found")

client = genai.Client(api_key=api_key)

print("Testing Gemini image generation...")

try:
    response = client.models.generate_content(
        model="gemini-3.1-flash-image",
        contents="Create a simple image of a luxury cosmetic cream jar on a white background.",
    )

    print("Gemini responded!")

    for part in response.candidates[0].content.parts:
        if part.inline_data:
            with open("test_image.png", "wb") as f:
                f.write(part.inline_data.data)

            print("SUCCESS!")
            print("Saved: test_image.png")

        elif part.text:
            print("TEXT RESPONSE:")
            print(part.text)

except Exception as e:
    print("\nERROR:")
    print(type(e).__name__)
    print(e)