from typing import List, Optional, Any, Union
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.callbacks import CallbackManagerForLLMRun
from pydantic import Field
from g4f.client import Client


class G4FChatModel(BaseChatModel):
    """Custom LangChain wrapper for g4f.Client"""
    
    model: str = Field(default="deepseek-v3")
    temperature: float = Field(default=0.0)
    web_search: bool = Field(default=False)
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate chat completion using g4f client."""
        
        # Convert LangChain messages to g4f format
        g4f_messages = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                g4f_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                g4f_messages.append({"role": "assistant", "content": msg.content})
            elif isinstance(msg, SystemMessage):
                g4f_messages.append({"role": "system", "content": msg.content})
            else:
                g4f_messages.append({"role": "user", "content": str(msg.content)})
        
        # Call g4f client
        client = Client()
        response = client.chat.completions.create(
            model=self.model,
            messages=g4f_messages,
            web_search=self.web_search,
            temperature=self.temperature
        )
        
        # Convert response to LangChain format
        content = response.choices[0].message.content
        message = AIMessage(content=content)
        generation = ChatGeneration(message=message)
        
        return ChatResult(generations=[generation])
    
    def invoke(self, input: Union[str, List[BaseMessage], dict], config=None, **kwargs) -> AIMessage:
        """Invoke method for Runnable compatibility."""
        # Handle different input types
        if isinstance(input, str):
            messages = [HumanMessage(content=input)]
        elif isinstance(input, dict):
            # Assume it's a formatted prompt from PromptTemplate
            if "text" in input:
                messages = [HumanMessage(content=input["text"])]
            else:
                # Try to convert dict values to string
                messages = [HumanMessage(content=str(input))]
        elif isinstance(input, list):
            messages = input
        else:
            messages = [HumanMessage(content=str(input))]
        
        result = self._generate(messages)
        return result.generations[0].message
    
    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "g4f-chat"
    
    def _identifying_params(self):
        """Get identifying parameters."""
        return {
            "model": self.model,
            "temperature": self.temperature,
            "web_search": self.web_search
        }
    
    def with_structured_output(self, schema):
        """Return a wrapper that outputs structured data according to schema."""
        from langchain_core.output_parsers import PydanticOutputParser
        from langchain_core.runnables import RunnablePassthrough
        
        # Create a parser for the schema
        parser = PydanticOutputParser(pydantic_object=schema)
        
        # Return a chain that includes format instructions and parsing
        def structured_invoke(input_text):
            # Add format instructions to the prompt
            format_instructions = parser.get_format_instructions()
            enhanced_prompt = f"{input_text}\n\n{format_instructions}\n\nOutput only valid JSON."
            
            # Generate response
            messages = [HumanMessage(content=enhanced_prompt)]
            result = self._generate(messages)
            
            # Parse the output
            raw_output = result.generations[0].message.content
            
            # Try to extract JSON if wrapped in markdown
            import re
            json_match = re.search(r'```json\s*(.*?)\s*```', raw_output, re.DOTALL)
            if json_match:
                raw_output = json_match.group(1)
            
            # Parse with the parser
            return parser.parse(raw_output)
        
        # Create a simple callable wrapper
        class StructuredOutputWrapper:
            def __init__(self, invoke_fn):
                self.invoke_fn = invoke_fn
            
            def invoke(self, input_val):
                return self.invoke_fn(input_val)
        
        return StructuredOutputWrapper(structured_invoke)

