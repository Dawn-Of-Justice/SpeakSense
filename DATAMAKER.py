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
                temperature=0.6,
                max_completion_tokens=4096,
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
    for i in range(332):
        # Initialize the generator
        try:
            generator = GroqGenerator(api_key="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")
            response = generator.generate_response(
                # prompt="So generate a lot of examples like the above, in this format \n\"sample1: <text>\"\nNo need to mention if its direct or indirect, only the above format should be followed strictly",
                # system_message="You are an AI Agent who is assigned with the task to make text outputs to train a model for context awareness task, basically the model will be trained for classifying if the model got addressed or not, like it would be used with an AI powered robot mostly, like it will listen to everything and when it is addressed only it will detect him, and start responding,\n\nLike Example,\n\nDirect Addressal:\nSample1: Hey Robot, can you tell me about the weather today?\nSample2: Hey buddy, tell me a joke!\n\nIndirect Addressal:\nSample1: So today we are talking about AI in Romantic Relationship, so what do you think Alexa\nSample2: I’ve been thinking about this a lot… what do you think?\nSample3: So, this is where we’re at… what should be done next?\nSample4: So, this is where we’re at… what should be done next?\n\nNot only this, I need samples of which talks about a random topic and then use this indirect addressals and direct addressals\n\nDirect Addressal:\nSample1: So we’ve been discussing the best sci-fi movies of all time. Hey Robot, do you have a favorite?\nSample2: We’re talking about AI in education, and I wanted to ask you directly, Alexa, do you think AI tutors can replace human teachers?\nSample3: I was looking into how human languages evolve over time. Words change meaning, slang becomes standard, and even grammar rules shift. But what about AI? If we train you on new kinds of speech patterns, will you evolve in the same way? Robot, what’s your take—do you think AI language models will ever develop their own natural dialects?\n\nIndirect Addressal:\nSample1: I was reading about the Mars rover mission today. They found some interesting rock samples... what do you make of that?\nSample2: I read an article about the history of human language. It’s amazing how it evolved over time... would you say AI can ever truly grasp language like we do?\nSample3: I was reading about AI-generated music today, and it's fascinating. Some people say AI can never truly be creative because it lacks emotions, but others argue that creativity is just a pattern-based process, and AI can replicate that. I mean, if we can't really define creativity ourselves, how can we say AI isn’t capable of it? I wonder how you’d look at it… Would AI creativity ever be as valuable as human creativity?"
                prompt = "So generate a lot of examples like the above and try to make it different starting phrases, in this format \n\"Sample1: <text>\"\nNo need to mention if it's direct or indirect, only the above format should be followed strictly.",
                system_message = "You are an AI Agent who is assigned with the task to make text outputs to train a model for context awareness. The model will be trained to classify different types of conversations. However, in this case, you must generate examples **without any direct or indirect addressal to an AI, robot, or virtual assistant**. The samples should be completely natural, covering a variety of topics such as technology, history, philosophy, science, and daily life. **Ensure that all examples sound like organic conversations between humans and do not involve AI in any way.**\n\nLike Example,\n\nSample1: I was reading about the Mars rover mission today. They found some interesting rock samples. The images they sent back look incredible!\nSample2: I read an article about the history of human language. It’s amazing how it evolved over time. I wonder if language is still changing at the same pace today.\nSample3: I was reading about AI-generated music today, and it's fascinating. Some people say AI can never truly be creative because it lacks emotions, but others argue that creativity is just a pattern-based process, and humans follow similar patterns.\nSample4: We were talking about space exploration, and someone mentioned how the Voyager probe is still sending data back. That’s just incredible.\nSample5: The other day, I was watching a documentary about deep-sea creatures. It’s amazing how little we know about life in the ocean compared to space.\nSample6: The Renaissance period was such a fascinating time in history. The level of art, science, and culture that flourished back then still influences us today.\nSample7: I read a study on how bilingual people process information differently. It seems like speaking multiple languages actually shapes the way we think.\nSample8: It’s interesting how some people believe time is just a human-made concept, while others argue that it's a fundamental part of the universe itself.\nSample9: There’s been a lot of debate about whether social media helps people connect or isolates them even more. It’s hard to say for sure.\nSample10: I recently started learning about quantum mechanics, and it’s blowing my mind. The idea that particles can exist in multiple states at once is hard to wrap my head around."

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