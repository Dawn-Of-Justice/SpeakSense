import os
from groq import Groq
from datetime import datetime
import json

class GroqGenerator:
    def __init__(self, api_key=None, output_dir="groq_outputs"):
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
                temperature=0.74,
                max_completion_tokens=4790,
                top_p=0.95,
            )
            
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return None
            
    def save_response(self, prompt, response, metadata=None):
        """
        Save response to a text file with metadata
        
        Args:
            prompt (str): Original prompt
            response (str): Generated response
            metadata (dict): Optional additional metadata
        """
        if response is None:
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"response_{timestamp}.txt"
        filepath = os.path.join(self.output_dir, filename)
        
        # Create data structure for saving
        data = {
            "timestamp": timestamp,
            "prompt": prompt,
            "response": response,
            "model": self.model
        }
        
        if metadata:
            data.update(metadata)
            
        # Save as structured text file
        with open(filepath, 'w', encoding='utf-8') as f:
            # f.write("=== METADATA ===\n")
            # f.write(json.dumps(data, indent=2))
            # f.write("\n=== CONTENT ===\n")
            f.write(response)
            
        return filepath
        
    def split_for_training(self, filepath, max_chars=1000):
        """
        Split a saved response into training samples
        
        Args:
            filepath (str): Path to saved response file
            max_chars (int): Maximum characters per sample
            
        Returns:
            list: List of training samples
        """
        samples = []
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            # Extract just the response part after === CONTENT ===
            response = content.split("=== CONTENT ===\n")[1]
            
            # Split into chunks
            for i in range(0, len(response), max_chars):
                sample = response[i:i + max_chars]
                samples.append(sample)
                
        return samples
        
    def set_model(self, model_name):
        """
        Set the model to use for generation
        
        Args:
            model_name (str): Name of the model to use
        """
        self.model = model_name

# Example usage
if __name__ == "__main__":
    for i in range(217):
        # Initialize the generator
        try:
            generator = GroqGenerator(api_key="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")
            response = generator.generate_response(
                # prompt = """Generate the samples that are addressing the robot. Follow the format strictly. Format is given below.
                #             "Sample1: <text min 500 words>"
                #             Do not refer to the robot all the time, just need to make samples that are addressing the robot.
                #             Make at least 20 Samples.Do not include anything other than the format given above.
                #             Also reduce the use of same starting words try to create different cases.
                #             IMPORTANT: Reduce I'm, I'll, I've. Use different Starting Words other than these. And it's general for all the words dont't repeat same words in starting.
                #             """,
                # system_message ="""
                # You are an assistant who generates the samples for training a model that classifies whether a sentence spoken by a human addresses the robot or not.
                # Your samples should be diverse, natural, realistic, and unique. Use different forms of addressing, and also the sentences should be meaningful with some good length."""
                prompt = """Generate the samples that are not addressing the robot. Follow the format strictly. Format is given below.
                            "Sample1: <text min 500 words>"
                            Need to make samples that are not addressing the robot.
                            Make at least 20 Samples.Do not include anything other than the format given above.
                            Also reduce the use of same starting words try to create different cases.""",
                system_message = """You are an assistant who generates the samples for training a model that classifies whether a sentence spoken by a human addresses the robot or not.
                                    Your samples should be diverse, natural, realistic, and unique. Use different forms of addressing, and also the sentences should be meaningful with some good length."""

            )
            
            if response:
                # Save with some metadata
                filepath = generator.save_response(
                    prompt="So generate a lot of examples like the above, in this format \n\"sample1: <text>\"\nNo need to mention if its direct or indirect, only the above format should be followed strictly",
                    response=response,
                    metadata={"category": "science"}
                )
                
                # Split into training samples
                samples = generator.split_for_training(filepath)
                # print(f"\nGenerated {len(samples)} training samples for: {prompt}")
                # print(f"First sample: {samples[0][:100]}...")
                
        except ValueError as e:
            print(f"Error: {str(e)}")
        except Exception as e:
            print(f"Unexpected error: {str(e)}")