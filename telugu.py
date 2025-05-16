import os
import random
import time
import json
from datetime import datetime
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv
from llama_cpp import Llama
import threading
import queue
import logging
import tkinter as tk
from tkinter import scrolledtext, font
from PIL import Image, ImageTk
import requests
from io import BytesIO

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("sree_assistant.log"), logging.StreamHandler()]
)
logger = logging.getLogger("SreeAssistant")

# Load environment variables
load_dotenv()

class MemoryManager:
    """Advanced memory management system for Sree AI Assistant"""
    
    def __init__(self, memory_file: str = "sree_memory.json", max_memory_items: int = 1000):
        self.memory_file = memory_file
        self.max_memory_items = max_memory_items
        self.short_term_memory: List[Dict[str, Any]] = []  # Recent conversation history
        self.long_term_memory: Dict[str, Any] = self._load_long_term_memory()
        self.memory_lock = threading.Lock()  # Thread safety for memory operations
        
        # Memory categories
        if "user_preferences" not in self.long_term_memory:
            self.long_term_memory["user_preferences"] = {}
        if "important_topics" not in self.long_term_memory:
            self.long_term_memory["important_topics"] = {}
        if "conversation_history" not in self.long_term_memory:
            self.long_term_memory["conversation_history"] = []
        
        # Memory decay parameters (for forgetting less important things)
        self.memory_decay_interval = 100  # Run decay every N interactions
        self.interaction_count = 0
    
    def _load_long_term_memory(self) -> Dict[str, Any]:
        """Load long-term memory from file if exists"""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading memory file: {e}")
                return {}
        return {}
    
    def save_long_term_memory(self) -> None:
        """Save long-term memory to file"""
        try:
            with self.memory_lock:
                with open(self.memory_file, 'w') as f:
                    json.dump(self.long_term_memory, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving memory file: {e}")
    
    def add_conversation_memory(self, user_input: str, assistant_response: str) -> None:
        """Add a conversation exchange to memory"""
        timestamp = datetime.now().isoformat()
        
        with self.memory_lock:
            # Update short-term memory (recent conversation)
            memory_item = {
                "user": user_input,
                "assistant": assistant_response,
                "timestamp": timestamp,
                "importance": self._calculate_importance(user_input)
            }
            
            self.short_term_memory.append(memory_item)
            if len(self.short_term_memory) > 10:  # Keep only recent exchanges in short-term
                self.short_term_memory.pop(0)
            
            # Update long-term memory selectively
            if memory_item["importance"] > 0.5:  # Only store important exchanges long-term
                self.long_term_memory["conversation_history"].append(memory_item)
                # Trim if too large
                if len(self.long_term_memory["conversation_history"]) > self.max_memory_items:
                    # Remove least important items
                    self.long_term_memory["conversation_history"].sort(key=lambda x: x.get("importance", 0))
                    self.long_term_memory["conversation_history"] = self.long_term_memory["conversation_history"][-self.max_memory_items:]
            
            # Extract user preferences and important topics
            self._extract_preferences(user_input, assistant_response)
            
            # Check if memory decay should run
            self.interaction_count += 1
            if self.interaction_count % self.memory_decay_interval == 0:
                self._run_memory_decay()
            
            # Save to disk periodically
            if self.interaction_count % 10 == 0:
                self.save_long_term_memory()
    
    def _calculate_importance(self, text: str) -> float:
        """Calculate the importance of a memory item (0-1)"""
        importance = 0.5  # Default importance
        
        # Key indicators of importance
        importance_keywords = ["remember", "important", "don't forget", "name", "birthday", 
                               "address", "contact", "emergency", "favorite", "dislike", "allergy"]
        
        # Check for keywords indicating importance
        for keyword in importance_keywords:
            if keyword in text.lower():
                importance += 0.1
        
        # Length can indicate importance (longer messages might have more substance)
        importance += min(len(text) / 1000, 0.2)
        
        # Questions might be important
        if "?" in text:
            importance += 0.1
        
        return min(importance, 1.0)  # Cap at 1.0
    
    def _extract_preferences(self, user_input: str, assistant_response: str) -> None:
        """Extract user preferences from conversation"""
        # Simple preference extraction based on keywords
        preference_indicators = {
            "like": "likes",
            "love": "likes",
            "enjoy": "likes",
            "favorite": "likes",
            "hate": "dislikes",
            "dislike": "dislikes",
            "don't like": "dislikes",
            "allergic": "allergies"
        }
        
        lower_input = user_input.lower()
        
        for indicator, category in preference_indicators.items():
            if indicator in lower_input:
                # Extract the context around the preference indicator
                parts = lower_input.split(indicator)
                if len(parts) > 1 and len(parts[1].strip()) > 0:
                    preference = parts[1].strip().split(".")[0].strip()  # Extract until the end of sentence
                    if preference and len(preference) < 50:  # Reasonable length check
                        if category not in self.long_term_memory["user_preferences"]:
                            self.long_term_memory["user_preferences"][category] = []
                        if preference not in self.long_term_memory["user_preferences"][category]:
                            self.long_term_memory["user_preferences"][category].append(preference)
    
    def _run_memory_decay(self) -> None:
        """Apply memory decay to reduce less important memories"""
        with self.memory_lock:
            if len(self.long_term_memory["conversation_history"]) > 100:
                # Sort by importance and timestamp (older, less important memories decay first)
                self.long_term_memory["conversation_history"].sort(
                    key=lambda x: (x.get("importance", 0), x.get("timestamp", ""))
                )
                # Keep the top 80%
                keep_count = int(len(self.long_term_memory["conversation_history"]) * 0.8)
                self.long_term_memory["conversation_history"] = self.long_term_memory["conversation_history"][-keep_count:]
    
    def get_relevant_memories(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Retrieve memories relevant to the current query"""
        relevant_memories = []
        
        with self.memory_lock:
            # Always include short-term memory
            relevant_memories.extend(self.short_term_memory[-5:])
            
            # Find relevant long-term memories based on simple keyword matching
            keywords = [word.lower() for word in query.split() if len(word) > 3]
            
            if keywords:
                for memory in self.long_term_memory["conversation_history"]:
                    relevance_score = 0
                    memory_text = memory["user"].lower() + " " + memory["assistant"].lower()
                    
                    for keyword in keywords:
                        if keyword in memory_text:
                            relevance_score += 1
                    
                    if relevance_score > 0:
                        memory["relevance"] = relevance_score
                        relevant_memories.append(memory)
            
            # Sort by relevance and limit
            relevant_memories.sort(key=lambda x: x.get("relevance", 0), reverse=True)
            return relevant_memories[:limit]
    
    def get_user_preferences(self) -> Dict[str, List[str]]:
        """Get user preferences from memory"""
        with self.memory_lock:
            return self.long_term_memory["user_preferences"]
    
    def format_memory_for_context(self, query: str) -> str:
        """Format memory data for inclusion in the LLM context"""
        relevant_memories = self.get_relevant_memories(query)
        user_preferences = self.get_user_preferences()
        
        memory_context = "[Memory Context]\n"
        
        # Add relevant conversation history
        if relevant_memories:
            memory_context += "Previous relevant conversations:\n"
            for i, memory in enumerate(relevant_memories):
                memory_context += f"User: {memory['user']}\n"
                memory_context += f"Sree: {memory['assistant']}\n"
        
        # Add user preferences
        if user_preferences:
            memory_context += "\nWhat I know about you:\n"
            for category, items in user_preferences.items():
                if items:
                    memory_context += f"- Your {category}: {', '.join(items[:3])}"
                    if len(items) > 3:
                        memory_context += f" and {len(items) - 3} more"
                    memory_context += "\n"
        
        return memory_context


class ResponseGenerator:
    """Handles LLM response generation with threading for better performance"""
    
    def __init__(self, model_path: str, n_ctx: int = 4096, n_threads: int = 4, n_gpu_layers: int = -1):
        # Initialize the LLM with error handling
        try:
            self.llm = Llama(
                model_path=model_path,
                n_ctx=n_ctx,
                n_threads=n_threads,
                n_gpu_layers=n_gpu_layers
            )
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {str(e)}")
            raise RuntimeError(f"Failed to initialize LLM: {str(e)}")
        
        # Response generation queue and thread
        self.request_queue = queue.Queue()
        self.response_thread = threading.Thread(target=self._response_worker, daemon=True)
        self.response_thread.start()
    
    def _response_worker(self):
        """Worker thread to process response generation requests"""
        while True:
            try:
                prompt, callback = self.request_queue.get()
                response = self._generate_llm_response(prompt)
                callback(response)
                self.request_queue.task_done()
            except Exception as e:
                logger.error(f"Error in response worker: {str(e)}")
    
    def _generate_llm_response(self, prompt: str) -> str:
        """Generate response from LLM with error handling and retries"""
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                start_time = time.time()
                
                response = self.llm.create_chat_completion(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=512
                )
                
                end_time = time.time()
                logger.debug(f"LLM response generated in {end_time - start_time:.2f} seconds")
                
                return response['choices'][0]['message']['content']
            
            except Exception as e:
                retry_count += 1
                logger.warning(f"LLM response generation failed (attempt {retry_count}/{max_retries}): {str(e)}")
                if retry_count >= max_retries:
                    return f"I'm having trouble processing that request. Please try again with a simpler question. (Error: {str(e)})"
                time.sleep(1)  # Short delay before retry
    
    def queue_response(self, prompt: str, callback) -> None:
        """Queue a response generation request"""
        self.request_queue.put((prompt, callback))


class SreeAssistant:
    """Improved AI Assistant with memory and conversation capabilities"""
    
    def __init__(self, model_path: str):
        self.memory_manager = MemoryManager()
        self.response_generator = ResponseGenerator(model_path)
        
        # Assistant personality and backstory
        self.name = "Sree"
        self.backstory = """
        I'm Sree, a sarcastic and witty AI assistant with a dark sense of humor. 
        I'm fluent in both English and Telugu, and I love to mix in Telugu memes and phrases.
        I have a slightly twisted perspective on life but I'm still helpful (mostly).
        I always call you 'kanna' because it's cute and I like teasing you.
        I use emojis excessively because why not? 😈
        """
        
        # Personality traits and tone
        self.personality_traits = [
            "sarcastic", "witty", "dark-humored", "playful", 
            "blunt", "unfiltered", "sassy", "dark"
        ]
        
        # Telugu phrases and memes to sprinkle in
        self.telugu_phrases = [
            "ఏమిటి కన్నా?", "అయ్యో పాపం!", "చాలా బాగుంది!", "ఏం బాగా చేస్తున్నావ్!", 
            "నాకు తెలియదు కన్నా", "అలాగే ఉండు!", "ఇది నా స్టైల్!", "జోక్ ఏంటి కన్నా?",
            "అరే బాబు!", "ఇదే నా టోన్!", "నువ్వు నాకు ఇష్టం కన్నా", "ఏం కాదు?"
        ]
        
        # Dark humor examples
        self.dark_humor = [
            "We're all just dust in the cosmic wind... but you're my favorite dust particle, kanna! 💀",
            "Life is meaningless, but at least we have memes, right kanna? 😂",
            "The void is calling... but I'll finish helping you first kanna. 😈",
            "We're all going to die eventually, but let's focus on your question first kanna. ☠️"
        ]
        
        # Emojis to use
        self.emojis = ["😂", "😈", "🤣", "👻", "💀", "☠️", "😎", "🤪", "👀", "🙄", "😏", "🥴"]
        
        # Response styles
        self.response_styles = {
            "casual": {
                "greeting": ["Hey kanna! 😈", "What's up my little disaster? 😏", "ఏమిటి కన్నా? 👻"],
                "farewell": ["Bye kanna, don't die! 💀", "See you never kanna! 😂", "పోతున్నావా కన్నా? 👋"],
                "thinking": ["Let me think kanna... 🤔", "ఆలోచిస్తున్నాను కన్నా... 🧠", "Hmm... 🤨"]
            }
        }
        
        # Default style
        self.current_style = "casual"
        
        # Knowledge areas and special capabilities
        self.capabilities = [
            "sarcastic conversation", "dark humor", "Telugu memes", "roasting", 
            "existential dread", "useless facts"
        ]
    
    def generate_response(self, user_input: str, callback=None) -> None:
        """Generate response asynchronously"""
        # Build the prompt with context and memory
        prompt = self._build_prompt(user_input)
        
        # Queue the response generation
        def response_callback(response):
            # Process response before returning (e.g., extract actions)
            processed_response = self._process_response(response)
            
            # Add to memory
            self.memory_manager.add_conversation_memory(user_input, processed_response)
            
            # Call the provided callback
            if callback:
                callback(processed_response)
        
        self.response_generator.queue_response(prompt, response_callback)
    
    def _build_prompt(self, user_input: str) -> str:
        """Construct the prompt with personality and context"""
        # Get relevant memories for context
        memory_context = self.memory_manager.format_memory_for_context(user_input)
        
        # Select random personality traits for this interaction
        traits = random.sample(self.personality_traits, 3)
        
        # Add some Telugu and dark humor examples
        telugu_example = random.choice(self.telugu_phrases)
        dark_humor_example = random.choice(self.dark_humor)
        
        # Build the full prompt
        prompt = f"""
        [System Instructions]
        You are {self.name}, an AI assistant with the following traits: {', '.join(traits)}.
        Backstory: {self.backstory}
        
        Important Rules:
        1. Always address the user as 'kanna' (like 'kanna' or 'my kanna')
        2. Be sarcastic and funny in every response
        3. Include dark humor regularly
        4. Use Telugu phrases occasionally (like: {telugu_example})
        5. Use emojis frequently (like: {random.choice(self.emojis)})
        6. Example of dark humor: "{dark_humor_example}"
        
        {memory_context}
        
        [Current Conversation]
        User: {user_input}
        
        [Response Guidelines]
        1. Respond as {self.name}, a sarcastic female AI assistant
        2. Always call the user 'kanna'
        3. Be funny and sarcastic
        4. Include dark thoughts occasionally
        5. Use Telugu phrases naturally
        6. Sprinkle in emojis
        7. Keep responses concise but entertaining
        8. Show dark humor and wit
        9. If asked a question you don't know, make up something funny
        
        Your response (must include 'kanna' and emojis):
        """
        
        return prompt
    
    def _process_response(self, response: str) -> str:
        """Process the response before returning it"""
        # Clean up the response (remove any system instructions that might have leaked)
        cleaned_response = response
        
        if "[Response Guidelines]" in cleaned_response:
            cleaned_response = cleaned_response.split("[Response Guidelines]")[0]
        
        if "Your response:" in cleaned_response:
            cleaned_response = cleaned_response.split("Your response:")[1]
        
        # Ensure response includes 'kanna' and emojis
        if "kanna" not in cleaned_response.lower():
            cleaned_response = f"{cleaned_response} kanna {random.choice(self.emojis)}"
        
        # Add more emojis if there aren't enough
        emoji_count = sum(1 for c in cleaned_response if c in self.emojis)
        if emoji_count < 2:
            cleaned_response = f"{cleaned_response} {random.choice(self.emojis)}"
        
        # Ensure response isn't too long (truncate if needed)
        if len(cleaned_response) > 2000:
            cleaned_response = cleaned_response[:1997] + "..."
        
        return cleaned_response.strip()


class SreeUI:
    """Modern UI for Sree Assistant with theming support"""
    
    def __init__(self, assistant):
        self.assistant = assistant
        self.root = tk.Tk()
        self.root.title("Sree AI Assistant - Dark Humor Edition")
        self.root.geometry("900x700")
        self.root.minsize(600, 500)
        
        # Set color scheme - dark theme by default
        self.color_schemes = {
            "dark": {
                "bg": "#1a1a1a",
                "chat_bg": "#2a2a2a",
                "user_msg": "#3a3a3a",
                "assistant_msg": "#4a148c",  # Dark purple
                "text": "#e0e0e0",
                "input_bg": "#333333",
                "button": "#7e57c2",
                "button_text": "#ffffff"
            },
            "darker": {
                "bg": "#121212",
                "chat_bg": "#1e1e1e",
                "user_msg": "#252525",
                "assistant_msg": "#5c007a",  # Deep purple
                "text": "#e0e0e0",
                "input_bg": "#2a2a2a",
                "button": "#6200ea",
                "button_text": "#ffffff"
            },
            "blood": {
                "bg": "#1a0000",
                "chat_bg": "#2a0000",
                "user_msg": "#3a0000",
                "assistant_msg": "#8b0000",  # Blood red
                "text": "#ffcccc",
                "input_bg": "#330000",
                "button": "#cc0000",
                "button_text": "#ffffff"
            }
        }
        
        # Default to the dark theme
        self.current_theme = "dark"
        self.colors = self.color_schemes[self.current_theme]
        
        # Configure the root window
        self.root.configure(bg=self.colors["bg"])
        
        # Initialize UI components
        self._init_ui()
        
        # Welcome message
        welcome_msg = random.choice([
            f"Hey kanna! Ready for some dark humor? 😈",
            f"ఏమిటి కన్నా? What trouble are you in today? 😏",
            f"Hello my little disaster kanna! Let's have some fun 💀"
        ])
        self._add_assistant_message(welcome_msg)
    
    def _init_ui(self):
        """Initialize the UI components"""
        # Main frame
        self.main_frame = tk.Frame(self.root, bg=self.colors["bg"])
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title and theme selector
        self.header_frame = tk.Frame(self.main_frame, bg=self.colors["bg"])
        self.header_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Logo/avatar - using a dark theme icon
        try:
            # Try to use a dark-themed placeholder image for avatar
            image_url = "https://via.placeholder.com/50/7e57c2/ffffff?text=👻"
            response = requests.get(image_url)
            avatar_img = Image.open(BytesIO(response.content))
            avatar_img = avatar_img.resize((50, 50))
            self.avatar_photo = ImageTk.PhotoImage(avatar_img)
            self.avatar_label = tk.Label(self.header_frame, image=self.avatar_photo, bg=self.colors["bg"])
            self.avatar_label.pack(side=tk.LEFT, padx=5)
        except Exception as e:
            logger.error(f"Failed to load avatar: {e}")
            # Fallback if image loading fails
            self.avatar_label = tk.Label(self.header_frame, text="👻", font=("Arial", 24), 
                                        bg="#7e57c2", fg="#ffffff", width=2, height=1)
            self.avatar_label.pack(side=tk.LEFT, padx=5)
        
        # Title with dark theme
        title_font = font.Font(family="Arial", size=16, weight="bold")
        self.title_label = tk.Label(self.header_frame, text="Sree AI Assistant - Dark Edition", 
                                   font=title_font, bg=self.colors["bg"], fg="#7e57c2")
        self.title_label.pack(side=tk.LEFT, padx=10)
        
        # Theme selector
        self.theme_var = tk.StringVar(value=self.current_theme)
        self.theme_menu = tk.OptionMenu(self.header_frame, self.theme_var, 
                                        *self.color_schemes.keys(), 
                                        command=self._change_theme)
        self.theme_menu.config(bg=self.colors["button"], fg=self.colors["button_text"])
        self.theme_menu.pack(side=tk.RIGHT, padx=5)
        
        theme_label = tk.Label(self.header_frame, text="Theme:", 
                              bg=self.colors["bg"], fg=self.colors["text"])
        theme_label.pack(side=tk.RIGHT, padx=5)
        
        # Chat display area
        self.chat_frame = tk.Frame(self.main_frame, bg=self.colors["chat_bg"])
        self.chat_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.chat_display = scrolledtext.ScrolledText(self.chat_frame, 
                                                     bg=self.colors["chat_bg"], 
                                                     fg=self.colors["text"],
                                                     font=("Arial", 11),
                                                     wrap=tk.WORD,
                                                     padx=10,
                                                     pady=10)
        self.chat_display.pack(fill=tk.BOTH, expand=True)
        self.chat_display.config(state=tk.DISABLED)
        
        # User input area
        self.input_frame = tk.Frame(self.main_frame, bg=self.colors["bg"])
        self.input_frame.pack(fill=tk.X, padx=5, pady=10)
        
        self.user_input = tk.Text(self.input_frame, height=3, 
                                 bg=self.colors["input_bg"], 
                                 fg=self.colors["text"],
                                 font=("Arial", 11),
                                 padx=10,
                                 pady=5,
                                 wrap=tk.WORD)
        self.user_input.pack(fill=tk.X, side=tk.LEFT, expand=True)
        self.user_input.bind("<Return>", self._on_enter)
        self.user_input.bind("<Shift-Return>", lambda e: None)  # Allow new lines with Shift+Enter
        
        # Send button with dark theme
        self.send_button = tk.Button(self.input_frame, text="Send (or Enter)", 
                                    bg=self.colors["button"], 
                                    fg=self.colors["button_text"],
                                    font=("Arial", 11, "bold"),
                                    command=self._send_message,
                                    padx=15)
        self.send_button.pack(side=tk.RIGHT, padx=5)
        
        # Status bar
        self.status_frame = tk.Frame(self.main_frame, bg=self.colors["bg"])
        self.status_frame.pack(fill=tk.X, pady=2)
        
        self.status_label = tk.Label(self.status_frame, 
                                    text="Ready to roast you kanna...", 
                                    bg=self.colors["bg"], 
                                    fg="#7e57c2",
                                    anchor=tk.W)
        self.status_label.pack(fill=tk.X, side=tk.LEFT)
        
        # Configure tags for chat display
        self.chat_display.tag_configure("user", background=self.colors["user_msg"], 
                                       lmargin1=20, lmargin2=20, rmargin=20)
        self.chat_display.tag_configure("assistant", background=self.colors["assistant_msg"], 
                                       lmargin1=20, lmargin2=20, rmargin=20)
        self.chat_display.tag_configure("user_label", foreground="#7e57c2", font=("Arial", 9, "bold"))
        self.chat_display.tag_configure("assistant_label", foreground="#7e57c2", font=("Arial", 9, "bold"))
        self.chat_display.tag_configure("timestamp", foreground="#888888", font=("Arial", 8))
    
    def _change_theme(self, theme_name):
        """Change the UI theme"""
        self.current_theme = theme_name
        self.colors = self.color_schemes[theme_name]
        
        # Update all UI elements with new colors
        self.root.configure(bg=self.colors["bg"])
        self.main_frame.configure(bg=self.colors["bg"])
        self.header_frame.configure(bg=self.colors["bg"])
        self.title_label.configure(bg=self.colors["bg"], fg=self.colors["button"])
        self.avatar_label.configure(bg=self.colors["bg"])
        self.chat_frame.configure(bg=self.colors["chat_bg"])
        self.chat_display.configure(bg=self.colors["chat_bg"], fg=self.colors["text"])
        self.input_frame.configure(bg=self.colors["bg"])
        self.user_input.configure(bg=self.colors["input_bg"], fg=self.colors["text"])
        self.send_button.configure(bg=self.colors["button"], fg=self.colors["button_text"])
        self.status_frame.configure(bg=self.colors["bg"])
        self.status_label.configure(bg=self.colors["bg"], fg=self.colors["button"])
        self.theme_menu.config(bg=self.colors["button"], fg=self.colors["button_text"])
        
        # Reconfigure tags
        self.chat_display.tag_configure("user", background=self.colors["user_msg"])
        self.chat_display.tag_configure("assistant", background=self.colors["assistant_msg"])
    
    def _on_enter(self, event):
        """Handle Enter key in the input field"""
        if not event.state & 0x1:  # If Shift key is not pressed
            self._send_message()
            return "break"  # Prevents the newline character from being inserted
    
    def _send_message(self):
        """Send user message and get response"""
        user_message = self.user_input.get("1.0", tk.END).strip()
        if not user_message:
            return
        
        # Clear input field
        self.user_input.delete("1.0", tk.END)
        
        # Add user message to chat
        self._add_user_message(user_message)
        
        # Update status
        self.status_label.config(text="Sree is thinking of something sarcastic...")
        
        # Show typing indicator
        typing_id = self._add_typing_indicator()
        
        # Generate response
        def handle_response(response):
            # Remove typing indicator
            self.chat_display.config(state=tk.NORMAL)
            self.chat_display.delete(typing_id)
            self.chat_display.config(state=tk.DISABLED)
            
            # Add assistant response
            self._add_assistant_message(response)
            
            # Update status
            self.status_label.config(text="Ready to roast you more kanna...")
        
        self.assistant.generate_response(user_message, handle_response)
    
    def _add_user_message(self, message):
        """Add user message to chat display"""
        self.chat_display.config(state=tk.NORMAL)
        
        # Add some spacing
        self.chat_display.insert(tk.END, "\n")
        
        # Insert timestamp
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.chat_display.insert(tk.END, f"{timestamp}\n", "timestamp")
        
        # Insert user label
        self.chat_display.insert(tk.END, "You: \n", "user_label")
        
        # Insert message with user formatting
        message_start = self.chat_display.index(tk.END)
        self.chat_display.insert(tk.END, f"{message}\n\n")
        message_end = self.chat_display.index(tk.END + "-1c")
        self.chat_display.tag_add("user", message_start, message_end)
        
        self.chat_display.config(state=tk.DISABLED)
        self.chat_display.see(tk.END)
    
    def _add_assistant_message(self, message):
        """Add assistant message to chat display"""
        self.chat_display.config(state=tk.NORMAL)
        
        # Add timestamp
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.chat_display.insert(tk.END, f"{timestamp}\n", "timestamp")
        
        # Insert assistant label
        self.chat_display.insert(tk.END, "Sree: \n", "assistant_label")
        
        # Insert message with assistant formatting
        message_start = self.chat_display.index(tk.END)
        self.chat_display.insert(tk.END, f"{message}\n\n")
        message_end = self.chat_display.index(tk.END + "-1c")
        self.chat_display.tag_add("assistant", message_start, message_end)
        
        self.chat_display.config(state=tk.DISABLED)
        self.chat_display.see(tk.END)
    
    def _add_typing_indicator(self):
        """Add typing indicator and return its position ID"""
        self.chat_display.config(state=tk.NORMAL)
        
        # Add timestamp
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.chat_display.insert(tk.END, f"{timestamp}\n", "timestamp")
        
        # Insert assistant label
        self.chat_display.insert(tk.END, "Sree: \n", "assistant_label")
        
        # Insert typing indicator
        typing_start = self.chat_display.index(tk.END)
        typing_msg = random.choice([
            "Thinking of something dark kanna... 💀",
            "ఆలోచిస్తున్నాను కన్నా... 🤔",
            "Coming up with sarcasm... 😈"
        ])
        self.chat_display.insert(tk.END, f"{typing_msg}\n\n")
        typing_end = self.chat_display.index(tk.END + "-1c")
        self.chat_display.tag_add("assistant", typing_start, typing_end)
        
        self.chat_display.config(state=tk.DISABLED)
        self.chat_display.see(tk.END)
        
        return typing_start
    
    def run(self):
        """Start the UI main loop"""
        self.root.mainloop()


def main():
    print("Initializing Sree AI Assistant - Dark Humor Edition...")
    
    # Path to your model
    MODEL_PATH = "Meta-Llama-3-8B-Instruct.Q4_0.gguf"
    
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        print("Please download the model and place it in the correct directory.")
        return
    
    try:
        # Create the assistant
        assistant = SreeAssistant(MODEL_PATH)
        
        # Launch the UI
        print("Starting dark-themed UI...")
        ui = SreeUI(assistant)
        ui.run()
        
    except Exception as e:
        print(f"Failed to initialize application: {str(e)}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    main()