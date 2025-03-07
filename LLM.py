import os
from groq import Groq

class AI:
    def __init__(self, api_key="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", output_dir="groq_outputs"):
        """
        Initialize the GroqGenerator with API key and output directory
        
        Args:
            api_key (str): Groq API key (defaults to environment variable if None)
            output_dir (str): Directory to save output files
        """
        # Use provided API key or fall back to environment variable
        self.api_key = api_key if api_key else os.environ.get("GROK_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not provided and not found in environment variables")
            
        self.client = Groq(api_key=self.api_key)
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Default model - can be changed via method
        self.model = "qwen-2.5-32b"
        
    def generate_response(self, prompt, system_message=None, max_tokens=1000):
        """
        Generate a response using Groq API
        
        Args:
            prompt (str): User prompt
            system_message (str): Optional system message for context
            max_tokens (int): Maximum tokens in response
            
        Returns:
            str: Generated response
        """
        messages = []
        
        if system_message:
            messages.append({"role": "system", "content": system_message})
            
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.6,
                max_completion_tokens=4096,
                top_p=0.95,
            )
            
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return None
        
if __name__ == "__main__":
    
    groq_ai = AI()
    response = groq_ai.generate_response(prompt="Input: The Eiffel Tower is one of the most iconic landmarks in the world, located in Paris, France. It was designed by Gustave Eiffel and completed in 1889 as the entrance arch for the 1889 Exposition Universelle (World’s Fair).What do you think.", 
    system_message =  "You are an AI model who gives to Live transcription message when you are addressed\n\ninput: I was reading about the use of AI in agriculture. What do you think,\noutput: <response>",
    )
    print(response)