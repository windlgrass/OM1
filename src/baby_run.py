"""
Baby Emotion Teaching Agent Runner
Standalone script - runs independently from OM1 runtime
"""

import asyncio
import json5
import logging
import signal
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class BabyRunner:
    """Runner for Baby Emotion Teaching Agent"""
    
    def __init__(self, config_path='config/baby.json5'):
        # Load config
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json5.load(f)
        
        self.api_key = self.config.get('api_key', 'openmind_free')
        self.api_url = "https://api.openmind.org/api/core/openai/chat/completions"
        
        teaching_config = self.config.get('teaching_config', {})
        self.model = teaching_config.get('model', 'gpt-4o')
        self.temperature = teaching_config.get('temperature', 1.2)
        self.max_tokens = teaching_config.get('max_tokens', 1000)
        self.turn_delay = teaching_config.get('turn_delay', 1.0)
        
        self.assistant_prompt = self.config.get('system_prompt_base', '')
        self.baby_prompt = self.config.get('baby_system_prompt', '')
        
        self.assistant_memory = []
        self.baby_memory = []
        self.turn_count = 0
        
        # Setup signal handler
        signal.signal(signal.SIGINT, self.signal_handler)
        
        logging.info("Baby Agent initialized")
    
    def signal_handler(self, sig, frame):
        """Handle Ctrl+C gracefully"""
        logging.info(f"\n\nStopping after {self.turn_count} turns")
        logging.info("Session ended")
        sys.exit(0)
    
    async def ask_api(self, system_prompt, messages, name):
        """Call API with streaming"""
        import aiohttp
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                *messages
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": True,
        }
        
        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
        }
        
        # Color output
        color = "\033[1;35m" if name == "Assistant" else "\033[1;32m"
        reset = "\033[0m"
        
        print(f"{color}{name}: {reset}", end="", flush=True)
        
        full_text = ""
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.api_url,
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    response.raise_for_status()
                    
                    async for line in response.content:
                        line_text = line.decode('utf-8').strip()
                        
                        if not line_text or not line_text.startswith('data: '):
                            continue
                        
                        data = line_text[6:]
                        if data.strip() == "[DONE]":
                            break
                        
                        try:
                            import json
                            delta = json.loads(data)["choices"][0]["delta"].get("content", "")
                            if delta:
                                full_text += delta
                                print(f"{color}{delta}{reset}", end="", flush=True)
                                await asyncio.sleep(0.05)
                        except Exception:
                            continue
        
        except Exception as e:
            logging.error(f"API error for {name}: {e}")
            return ""
        
        print()  # New line
        return full_text
    
    async def teaching_turn(self):
        """Execute one teaching turn"""
        self.turn_count += 1
        
      print(f"\n{'='*60}\n")
        
        try:
            # Assistant teaches
            assistant_msg = await self.ask_api(
                self.assistant_prompt,
                self.assistant_memory,
                "Assistant"
            )
            
            if assistant_msg:
                self.assistant_memory.append({
                    "role": "assistant",
                    "content": assistant_msg
                })
                self.baby_memory.append({
                    "role": "user",
                    "content": assistant_msg
                })
            
            await asyncio.sleep(self.turn_delay)
            
            # Baby responds
            baby_msg = await self.ask_api(
                self.baby_prompt,
                self.baby_memory,
                "Baby"
            )
            
            if baby_msg:
                self.baby_memory.append({
                    "role": "assistant",
                    "content": baby_msg
                })
                self.assistant_memory.append({
                    "role": "user",
                    "content": baby_msg
                })
            
            await asyncio.sleep(self.turn_delay)
            
            # Log progress
            if self.turn_count % 10 == 0:
                logging.info(f"Progress: {self.turn_count} turns completed")
        
        except Exception as e:
            logging.error(f"Error in teaching turn: {e}")
            raise
    
    async def run(self):
        """Main teaching loop"""
        logging.info("🤖 Starting Baby Emotion Teaching Agent")
        logging.info("Press Ctrl+C to stop")
        
        # Initial message
        initial_topic = self.config.get('teaching_config', {}).get(
            'initial_topic',
            'cảm xúc con người'
        )
        
        self.assistant_memory.append({
            "role": "user",
            "content": f"Baby, tôi sẽ dạy em về {initial_topic}."
        })
        
        try:
            while True:
                await self.teaching_turn()
        
        except KeyboardInterrupt:
            self.signal_handler(None, None)
        except Exception as e:
            logging.error(f"Error in teaching loop: {e}")
            raise


if __name__ == "__main__":
    # Check if config exists
    config_path = Path('config/baby.json5')
    
    if not config_path.exists():
        logging.error("Config file not found: config/baby.json5")
        logging.error("Please create the config file first")
        sys.exit(1)
    
    # Run Baby Agent
    runner = BabyRunner()
    asyncio.run(runner.run())
