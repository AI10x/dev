import google.generativeai as genai
import os

class GeminiGenerator:
    def __init__(self, api_key, system_instruction=None):
        """Initializes the GeminiGenerator."""
        if not api_key:
            raise ValueError("API key not provided")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            model_name='gemini-2.5-pro',
            system_instruction=system_instruction
        )

    def generate(self, prompt):
        """Generates text from a given prompt."""
        response = self.model.generate_content(prompt)
        return response.text

if __name__ == "__main__":
    # Fetch the API key from an environment variable
    api_key = os.getenv("GOOGLE_API_KEY")

    # Ensure the API key is set
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set")

    # Add a system instruction
    system_instruction = "You are a helpful assistant that provides concise and accurate answers."

    # Create an instance of the generator
    generator = GeminiGenerator(api_key=api_key, system_instruction=system_instruction)

    # Get a prompt from the user
    prompting = input("Prompt: ")

    # Generate and print the text
    generated_text = generator.generate(prompting)
    print(generated_text)
