import os
from groq import Groq
from datetime import datetime
import json
import time

system_prompt = "You are an AI assistant tasked with generating synthetic training data for an addressing classifier model. This model will determine whether speech is addressed to an AI assistant or not, replacing traditional wake word detection. \n\nYour task is to generate realistic speech samples that might occur in real-time conversations, including:\n1. Complete sentences (some addressed to AI, some not)\n2. Cut-off or incomplete sentences\n3. Short addressing phrases\n4. Sentences with occasional transcription errors or hallucinated words\n\nEach generated sample should be clearly follow the format\n\nADDRESSING SAMPLES should include:\n- Questions or statements clearly directed at an AI assistant\n- Phrases that seek information, opinions, or actions from the AI\n- Direct addresses that may or may not include \"hey\", \"hi\", or similar greetings\n- May include cut-off beginnings or endings as would occur in real-time transcription\n\nNOT ADDRESSING SAMPLES should include:\n- People talking to each other, not to the AI\n- Statements or questions that are part of a human-to-human conversation\n- Ambient speech not directed at any particular entity\n- May include cut-off beginnings or endings as would occur in real-time transcription\n\nFORMAT:\nSample1: [generated text sample]\nSample2: [generated text sample]\nSample3: [generated text sample]\n...\n\nEXAMPLES:\n\nADDRESSING:\n- Hey what's the weather going to be like tomorrow\n- The HR department is looking to update the employee handbook Could you help by compiling all the necessary information and drafting the new version of the handbook It would be great if you could also gather feedback from the employees to ensure the handbook is comprehensive and useful\n- I was reading about the future of AI in the workplace Do you think AI will replace human jobs\n- What do you think about the latest movie releases\n- Tell me about the history of artificial intelligence\n- Do you know how to make a good pasta sauce\n\nNOT ADDRESSING:\n- Ive been learning more about photography lately and its fascinating how a single photo can tell a whole story Ive been experimenting with different angles and lighting and its exciting to see how the same scene can look so different based on how you capture it\n-My sister called me yesterday to tell me about her new job She started working at a tech company and shes excited about the opportunities there She mentioned they have a great team and are developing some innovative software\n- Its been a while since we visited the museum I remember the last time we went we spent hours looking at the art exhibits Maybe we should plan a visit for this weekend Ive been wanting to see the new Picasso collection they just acquired\n- so you're doing this and yeah maybe if its done\n- I really enjoyed that book you recommended last week\n- My car needs to be taken in for service soon\n- C.C.C.C C.C.C.C C.C.C.C blurble\n\nADDITIONAL INSTRUCTIONS:\n- Include some samples with cut-off text (e.g., \"so you're doing this and yeah maybe if its done\")\n- Include some short addressing phrases (e.g., \"Hey\", \"Hi there\", \"Excuse me\")\n- Include some samples with nonsensical or hallucinated words (e.g., \"C.C.C.C\", \"blurble\")\n- Create a mix of formal and informal language styles\n- Include samples from various topics (technology, entertainment, daily life, etc.)\n- Ensure a balanced distribution of addressing and non-addressing samples"

prompt = "Generate 100 token long 20 samples of Addressing, this should be the worst case scenario like samples which might come in live transcriptions, always address at the end only, don't use can you, what do you think, say something else"
#  don't follow the same pattern of conversation given to u in the addressing examples, make new ones, diverse ones



class GroqGenerator:
    def __init__(self, api_key=None, output_dir="groq_outputs"):
        """
        Initialize the GroqGenerator with API key and output directory
        
        Args:
            api_key (str): Groq API key (defaults to environment variable if None)
            output_dir (str): Directory to save output files
        """
        # Use provided API key or fall back to environment variable
        # self.api_key = api_key
        self.api_key = api_key if api_key else os.environ.get("GROK_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not provided and not found in environment variables")
            
        self.client = Groq(api_key=self.api_key)
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Default model - can be changed via method
        self.model = "llama-3.3-70b-versatile"
        
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
                temperature=1,
                max_completion_tokens=1024,
                top_p=1,
                stop=None,
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
                prompt = prompt,
                system_message = system_prompt

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
            time.sleep(2)
        except ValueError as e:
            print(f"Error: {str(e)}")
        except Exception as e:
            print(f"Unexpected error: {str(e)}")