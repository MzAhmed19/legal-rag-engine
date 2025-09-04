import os
from groq import Groq
from dotenv import load_dotenv

# Load environment variables from your .env file
load_dotenv()

print("--- Testing Direct Connection to Groq API ---")

# 1. Check if the API key is loaded
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    print("❌ FATAL: GROQ_API_KEY not found in your .env file.")
else:
    print("✅ API key loaded successfully.")
    
    try:
        # 2. Initialize the Groq client
        client = Groq(api_key=api_key)
        print("✅ Groq client initialized.")
        
        # 3. Make a simple API call
        print("⏳Sending a simple test message to the llama3-8b-8192 model...")
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": "Explain the importance of low latency in LLMs in one sentence.",
                }
            ],
            model="llama3-70b-8192",
        )
        
        # 4. Print the result
        response_content = chat_completion.choices[0].message.content
        print("\n--- ✅ SUCCESS! ---")
        print(f"Groq API responded successfully:\n\n'{response_content}'")

    except Exception as e:
        print("\n--- ❌ FAILURE! ---")
        print(f"The direct API call failed. This confirms the issue is with the Groq service or your connection.")
        print(f"Error details: {e}")