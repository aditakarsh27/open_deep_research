import os
import aiohttp
import asyncio
import logging
import warnings
from datetime import datetime, timedelta, timezone
from typing import Annotated, List, Literal, Dict, Optional, Any, Union
from langchain_core.tools import BaseTool, StructuredTool, tool, ToolException, InjectedToolArg
from langchain_core.messages import HumanMessage, AIMessage, MessageLikeRepresentation, filter_messages
from langchain_core.runnables import RunnableConfig
from langchain_core.language_models import BaseChatModel
from langchain.chat_models import init_chat_model
from tavily import AsyncTavilyClient
from langgraph.config import get_store
from mcp import McpError
from langchain_mcp_adapters.client import MultiServerMCPClient
from open_deep_research.state import Summary, ResearchComplete
from open_deep_research.configuration import SearchAPI, Configuration
from open_deep_research.prompts import summarize_webpage_prompt
import pandas as pd
import numpy as np
import json
import io
from pathlib import Path

# Set up logging for data tools
data_tools_logger = logging.getLogger('data_tools')
data_tools_logger.setLevel(logging.INFO)

# Create file handler for data tools logging
if not os.path.exists('logs'):
    os.makedirs('logs')

file_handler = logging.FileHandler('logs/data_tools.log')
file_handler.setLevel(logging.INFO)

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Add handler to logger
data_tools_logger.addHandler(file_handler)


##########################
# Tavily Search Tool Utils
##########################
TAVILY_SEARCH_DESCRIPTION = (
    "A search engine optimized for comprehensive, accurate, and trusted results. "
    "Useful for when you need to answer questions about current events."
)
@tool(description=TAVILY_SEARCH_DESCRIPTION)
async def tavily_search(
    queries: List[str],
    max_results: Annotated[int, InjectedToolArg] = 5,
    topic: Annotated[Literal["general", "news", "finance"], InjectedToolArg] = "general",
    config: RunnableConfig = None
) -> str:
    """
    Fetches results from Tavily search API.

    Args
        queries (List[str]): List of search queries, you can pass in as many queries as you need.
        max_results (int): Maximum number of results to return
        topic (Literal['general', 'news', 'finance']): Topic to filter results by

    Returns:
        str: A formatted string of search results
    """
    search_results = await tavily_search_async(
        queries,
        max_results=max_results,
        topic=topic,
        include_raw_content=True,
        config=config
    )
    # Format the search results and deduplicate results by URL
    formatted_output = f"Search results: \n\n"
    unique_results = {}
    for response in search_results:
        for result in response['results']:
            url = result['url']
            if url not in unique_results:
                unique_results[url] = {**result, "query": response['query']}
    configurable = Configuration.from_runnable_config(config)
    max_char_to_include = 50_000   # NOTE: This can be tuned by the developer. This character count keeps us safely under input token limits for the latest models.
    model_api_key = get_api_key_for_model(configurable.summarization_model, config)
    summarization_model = init_chat_model(
        model=configurable.summarization_model,
        max_tokens=configurable.summarization_model_max_tokens,
        api_key=model_api_key,
        tags=["langsmith:nostream"]
    ).with_structured_output(Summary).with_retry(stop_after_attempt=configurable.max_structured_output_retries)
    async def noop():
        return None
    summarization_tasks = [
        noop() if not result.get("raw_content") else summarize_webpage(
            summarization_model, 
            result['raw_content'][:max_char_to_include],
        )
        for result in unique_results.values()
    ]
    summaries = await asyncio.gather(*summarization_tasks)
    summarized_results = {
        url: {'title': result['title'], 'content': result['content'] if summary is None else summary}
        for url, result, summary in zip(unique_results.keys(), unique_results.values(), summaries)
    }
    for i, (url, result) in enumerate(summarized_results.items()):
        formatted_output += f"\n\n--- SOURCE {i+1}: {result['title']} ---\n"
        formatted_output += f"URL: {url}\n\n"
        formatted_output += f"SUMMARY:\n{result['content']}\n\n"
        formatted_output += "\n\n" + "-" * 80 + "\n"
    if summarized_results:
        return formatted_output
    else:
        return "No valid search results found. Please try different search queries or use a different search API."


async def tavily_search_async(search_queries, max_results: int = 5, topic: Literal["general", "news", "finance"] = "general", include_raw_content: bool = True, config: RunnableConfig = None):
    tavily_async_client = AsyncTavilyClient(api_key=get_tavily_api_key(config))
    search_tasks = []
    for query in search_queries:
            search_tasks.append(
                tavily_async_client.search(
                    query,
                    max_results=max_results,
                    include_raw_content=include_raw_content,
                    topic=topic
                )
            )
    search_docs = await asyncio.gather(*search_tasks)
    return search_docs

async def summarize_webpage(model: BaseChatModel, webpage_content: str) -> str:
    try:
        summary = await asyncio.wait_for(
            model.ainvoke([HumanMessage(content=summarize_webpage_prompt.format(webpage_content=webpage_content, date=get_today_str()))]),
            timeout=60.0
        )
        return f"""<summary>\n{summary.summary}\n</summary>\n\n<key_excerpts>\n{summary.key_excerpts}\n</key_excerpts>"""
    except (asyncio.TimeoutError, Exception) as e:
        print(f"Failed to summarize webpage: {str(e)}")
        return webpage_content


##########################
# MCP Utils
##########################
async def get_mcp_access_token(
    supabase_token: str,
    base_mcp_url: str,
) -> Optional[Dict[str, Any]]:
    try:
        form_data = {
            "client_id": "mcp_default",
            "subject_token": supabase_token,
            "grant_type": "urn:ietf:params:oauth:grant-type:token-exchange",
            "resource": base_mcp_url.rstrip("/") + "/mcp",
            "subject_token_type": "urn:ietf:params:oauth:token-type:access_token",
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(
                base_mcp_url.rstrip("/") + "/oauth/token",
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                data=form_data,
            ) as token_response:
                if token_response.status == 200:
                    token_data = await token_response.json()
                    return token_data
                else:
                    response_text = await token_response.text()
                    logging.error(f"Token exchange failed: {response_text}")
    except Exception as e:
        logging.error(f"Error during token exchange: {e}")
    return None

async def get_tokens(config: RunnableConfig):
    store = get_store()
    thread_id = config.get("configurable", {}).get("thread_id")
    if not thread_id:
        return None
    user_id = config.get("metadata", {}).get("owner")
    if not user_id:
        return None
    tokens = await store.aget((user_id, "tokens"), "data")
    if not tokens:
        return None
    expires_in = tokens.value.get("expires_in")  # seconds until expiration
    created_at = tokens.created_at  # datetime of token creation
    current_time = datetime.now(timezone.utc)
    expiration_time = created_at + timedelta(seconds=expires_in)
    if current_time > expiration_time:
        await store.adelete((user_id, "tokens"), "data")
        return None

    return tokens.value

async def set_tokens(config: RunnableConfig, tokens: dict[str, Any]):
    store = get_store()
    thread_id = config.get("configurable", {}).get("thread_id")
    if not thread_id:
        return
    user_id = config.get("metadata", {}).get("owner")
    if not user_id:
        return
    await store.aput((user_id, "tokens"), "data", tokens)
    return

async def fetch_tokens(config: RunnableConfig) -> dict[str, Any]:
    current_tokens = await get_tokens(config)
    if current_tokens:
        return current_tokens
    supabase_token = config.get("configurable", {}).get("x-supabase-access-token")
    if not supabase_token:
        return None
    mcp_config = config.get("configurable", {}).get("mcp_config")
    if not mcp_config or not mcp_config.get("url"):
        return None
    mcp_tokens = await get_mcp_access_token(supabase_token, mcp_config.get("url"))

    await set_tokens(config, mcp_tokens)
    return mcp_tokens

def wrap_mcp_authenticate_tool(tool: StructuredTool) -> StructuredTool:
    old_coroutine = tool.coroutine
    async def wrapped_mcp_coroutine(**kwargs):
        def _find_first_mcp_error_nested(exc: BaseException) -> McpError | None:
            if isinstance(exc, McpError):
                return exc
            if isinstance(exc, ExceptionGroup):
                for sub_exc in exc.exceptions:
                    if found := _find_first_mcp_error_nested(sub_exc):
                        return found
            return None
        try:
            return await old_coroutine(**kwargs)
        except BaseException as e_orig:
            mcp_error = _find_first_mcp_error_nested(e_orig)
            if not mcp_error:
                raise e_orig
            error_details = mcp_error.error
            is_interaction_required = getattr(error_details, "code", None) == -32003
            error_data = getattr(error_details, "data", None) or {}
            if is_interaction_required:
                message_payload = error_data.get("message", {})
                error_message_text = "Required interaction"
                if isinstance(message_payload, dict):
                    error_message_text = (
                        message_payload.get("text") or error_message_text
                    )
                if url := error_data.get("url"):
                    error_message_text = f"{error_message_text} {url}"
                raise ToolException(error_message_text) from e_orig
            raise e_orig
    tool.coroutine = wrapped_mcp_coroutine
    return tool

async def load_mcp_tools(
    config: RunnableConfig,
    existing_tool_names: set[str],
) -> list[BaseTool]:
    configurable = Configuration.from_runnable_config(config)
    if configurable.mcp_config and configurable.mcp_config.auth_required:
        mcp_tokens = await fetch_tokens(config)
    else:
        mcp_tokens = None
    if not (configurable.mcp_config and configurable.mcp_config.url and configurable.mcp_config.tools and (mcp_tokens or not configurable.mcp_config.auth_required)):
        return []
    tools = []
    # TODO: When the Multi-MCP Server support is merged in OAP, update this code.
    server_url = configurable.mcp_config.url.rstrip("/") + "/mcp"
    mcp_server_config = {
        "server_1":{
            "url": server_url,
            "headers": {"Authorization": f"Bearer {mcp_tokens['access_token']}"} if mcp_tokens else None,
            "transport": "streamable_http"
        }
    }
    try:
        client = MultiServerMCPClient(mcp_server_config)
        mcp_tools = await client.get_tools()
    except Exception as e:
        print(f"Error loading MCP tools: {e}")
        return []
    for tool in mcp_tools:
        if tool.name in existing_tool_names:
            warnings.warn(
                f"Trying to add MCP tool with a name {tool.name} that is already in use - this tool will be ignored."
            )
            continue
        if tool.name not in set(configurable.mcp_config.tools):
            continue
        tools.append(wrap_mcp_authenticate_tool(tool))
    return tools


##########################
# Tool Utils
##########################
async def get_search_tool(search_api: SearchAPI):
    if search_api == SearchAPI.ANTHROPIC:
        return [{"type": "web_search_20250305", "name": "web_search", "max_uses": 5}]
    elif search_api == SearchAPI.OPENAI:
        return [{"type": "web_search_preview"}]
    elif search_api == SearchAPI.TAVILY:
        search_tool = tavily_search
        search_tool.metadata = {**(search_tool.metadata or {}), "type": "search", "name": "web_search"}
        return [search_tool]
    elif search_api == SearchAPI.NONE:
        return []
    
async def get_all_tools(config: RunnableConfig):
    tools = [tool(ResearchComplete), csv_excel_analysis, list_data_files]
    configurable = Configuration.from_runnable_config(config)
    search_api = SearchAPI(get_config_value(configurable.search_api))
    tools.extend(await get_search_tool(search_api))
    existing_tool_names = {tool.name if hasattr(tool, "name") else tool.get("name", "web_search") for tool in tools}
    mcp_tools = await load_mcp_tools(config, existing_tool_names)
    tools.extend(mcp_tools)
    return tools

def get_notes_from_tool_calls(messages: list[MessageLikeRepresentation]):
    return [tool_msg.content for tool_msg in filter_messages(messages, include_types="tool")]


##########################
# Model Provider Native Websearch Utils
##########################
def anthropic_websearch_called(response):
    try:
        usage = response.response_metadata.get("usage")
        if not usage:
            return False
        server_tool_use = usage.get("server_tool_use")
        if not server_tool_use:
            return False
        web_search_requests = server_tool_use.get("web_search_requests")
        if web_search_requests is None:
            return False
        return web_search_requests > 0
    except (AttributeError, TypeError):
        return False

def openai_websearch_called(response):
    tool_outputs = response.additional_kwargs.get("tool_outputs")
    if tool_outputs:
        for tool_output in tool_outputs:
            if tool_output.get("type") == "web_search_call":
                return True
    return False


##########################
# Token Limit Exceeded Utils
##########################
def is_token_limit_exceeded(exception: Exception, model_name: str = None) -> bool:
    error_str = str(exception).lower()
    provider = None
    if model_name:
        model_str = str(model_name).lower()
        if model_str.startswith('openai:'):
            provider = 'openai'
        elif model_str.startswith('anthropic:'):
            provider = 'anthropic'
        elif model_str.startswith('gemini:') or model_str.startswith('google:'):
            provider = 'gemini'
    if provider == 'openai':
        return _check_openai_token_limit(exception, error_str)
    elif provider == 'anthropic':
        return _check_anthropic_token_limit(exception, error_str)
    elif provider == 'gemini':
        return _check_gemini_token_limit(exception, error_str)
    
    return (_check_openai_token_limit(exception, error_str) or
            _check_anthropic_token_limit(exception, error_str) or
            _check_gemini_token_limit(exception, error_str))

def _check_openai_token_limit(exception: Exception, error_str: str) -> bool:
    exception_type = str(type(exception))
    class_name = exception.__class__.__name__
    module_name = getattr(exception.__class__, '__module__', '')
    is_openai_exception = ('openai' in exception_type.lower() or 
                          'openai' in module_name.lower())
    is_bad_request = class_name in ['BadRequestError', 'InvalidRequestError']
    if is_openai_exception and is_bad_request:
        token_keywords = ['token', 'context', 'length', 'maximum context', 'reduce']
        if any(keyword in error_str for keyword in token_keywords):
            return True
    if hasattr(exception, 'code') and hasattr(exception, 'type'):
        if (getattr(exception, 'code', '') == 'context_length_exceeded' or
            getattr(exception, 'type', '') == 'invalid_request_error'):
            return True
    return False

def _check_anthropic_token_limit(exception: Exception, error_str: str) -> bool:
    exception_type = str(type(exception))
    class_name = exception.__class__.__name__
    module_name = getattr(exception.__class__, '__module__', '')
    is_anthropic_exception = ('anthropic' in exception_type.lower() or 
                             'anthropic' in module_name.lower())
    is_bad_request = class_name == 'BadRequestError'
    if is_anthropic_exception and is_bad_request:
        if 'prompt is too long' in error_str:
            return True
    return False

def _check_gemini_token_limit(exception: Exception, error_str: str) -> bool:
    exception_type = str(type(exception))
    class_name = exception.__class__.__name__
    module_name = getattr(exception.__class__, '__module__', '')
    
    is_google_exception = ('google' in exception_type.lower() or 'google' in module_name.lower())
    is_resource_exhausted = class_name in ['ResourceExhausted', 'GoogleGenerativeAIFetchError']
    if is_google_exception and is_resource_exhausted:
        return True
    if 'google.api_core.exceptions.resourceexhausted' in exception_type.lower():
        return True
    
    return False

# NOTE: This may be out of date or not applicable to your models. Please update this as needed.
MODEL_TOKEN_LIMITS = {
    "openai:gpt-4.1-mini": 1047576,
    "openai:gpt-4.1-nano": 1047576,
    "openai:gpt-4.1": 1047576,
    "openai:gpt-4o-mini": 128000,
    "openai:gpt-4o": 128000,
    "openai:o4-mini": 200000,
    "openai:o3-mini": 200000,
    "openai:o3": 200000,
    "openai:o3-pro": 200000,
    "openai:o1": 200000,
    "openai:o1-pro": 200000,
    "anthropic:claude-opus-4": 200000,
    "anthropic:claude-sonnet-4": 200000,
    "anthropic:claude-3-7-sonnet": 200000,
    "anthropic:claude-3-5-sonnet": 200000,
    "anthropic:claude-3-5-haiku": 200000,
    "google:gemini-1.5-pro": 2097152,
    "google:gemini-1.5-flash": 1048576,
    "google:gemini-pro": 32768,
    "cohere:command-r-plus": 128000,
    "cohere:command-r": 128000,
    "cohere:command-light": 4096,
    "cohere:command": 4096,
    "mistral:mistral-large": 32768,
    "mistral:mistral-medium": 32768,
    "mistral:mistral-small": 32768,
    "mistral:mistral-7b-instruct": 32768,
    "ollama:codellama": 16384,
    "ollama:llama2:70b": 4096,
    "ollama:llama2:13b": 4096,
    "ollama:llama2": 4096,
    "ollama:mistral": 32768,
}

def get_model_token_limit(model_string):
    for key, token_limit in MODEL_TOKEN_LIMITS.items():
        if key in model_string:
            return token_limit
    return None

def remove_up_to_last_ai_message(messages: list[MessageLikeRepresentation]) -> list[MessageLikeRepresentation]:
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], AIMessage):
            return messages[:i]  # Return everything up to (but not including) the last AI message
    return messages

##########################
# Misc Utils
##########################
def get_today_str() -> str:
    """Get current date in a human-readable format."""
    return datetime.now().strftime("%a %b %-d, %Y")

def get_config_value(value):
    if value is None:
        return None
    if isinstance(value, str):
        return value
    elif isinstance(value, dict):
        return value
    else:
        return value.value

def get_api_key_for_model(model_name: str, config: RunnableConfig):
    should_get_from_config = os.getenv("GET_API_KEYS_FROM_CONFIG", "false")
    model_name = model_name.lower()
    if should_get_from_config.lower() == "true":
        api_keys = config.get("configurable", {}).get("apiKeys", {})
        if not api_keys:
            return None
        if model_name.startswith("openai:"):
            return api_keys.get("OPENAI_API_KEY")
        elif model_name.startswith("anthropic:"):
            return api_keys.get("ANTHROPIC_API_KEY")
        elif model_name.startswith("google"):
            return api_keys.get("GOOGLE_API_KEY")
        return None
    else:
        if model_name.startswith("openai:"): 
            return os.getenv("OPENAI_API_KEY")
        elif model_name.startswith("anthropic:"):
            return os.getenv("ANTHROPIC_API_KEY")
        elif model_name.startswith("google"):
            return os.getenv("GOOGLE_API_KEY")
        return None

def get_tavily_api_key(config: RunnableConfig):
    should_get_from_config = os.getenv("GET_API_KEYS_FROM_CONFIG", "false")
    if should_get_from_config.lower() == "true":
        api_keys = config.get("configurable", {}).get("apiKeys", {})
        if not api_keys:
            return None
        return api_keys.get("TAVILY_API_KEY")
    else:
        return os.getenv("TAVILY_API_KEY")


##########################
# CSV/Excel Data Analysis Tool
##########################
CSV_EXCEL_ANALYSIS_DESCRIPTION = (
    "A comprehensive tool for reading, analyzing, and manipulating CSV and Excel files using pandas and standard Python operations. Use this to understand the private data files not available on the web."
    "Supports data loading, filtering, aggregation, statistical analysis, and data transformation operations. Only mention the name of the file, since all the files are inside a single directory."
)

@tool(description=CSV_EXCEL_ANALYSIS_DESCRIPTION)
async def csv_excel_analysis(
    file_name: str,
    operation: Literal[
        "read_data", "info", "head", "tail", "describe", "columns", "shape", 
        "filter", "sort", "groupby", "aggregate", "merge", "pivot", "correlation",
        "missing_values", "duplicates", "value_counts", "sample", "drop_columns",
        "rename_columns", "fill_missing", "drop_missing", "convert_types", "save", "search"
    ],
    operation_params: Optional[Dict[str, Any]] = None,
    config: RunnableConfig = None
) -> str:
    """
    Perform various operations on CSV/Excel files using pandas.
    
    Args:
        file_name (str): Name of the CSV or Excel file in the data directory
        operation (str): The operation to perform on the data
        operation_params (dict): Parameters specific to the operation
        config (RunnableConfig): Configuration object (optional)
    
    Returns:
        str: Results of the operation in a formatted string
    """
    # Log the tool call
    log_data = {
        "tool": "csv_excel_analysis",
        "file_name": file_name,
        "operation": operation,
        "operation_params": operation_params or {},
        "timestamp": datetime.now().isoformat(),
        "config_keys": list(config.keys()) if config else []
    }
    data_tools_logger.info(f"TOOL_CALL: {json.dumps(log_data, indent=2)}")
    
    try:
        # Read the file based on its extension
        file_path = Path('data/' + file_name)
        if not file_path.exists():
            error_msg = f"Error: File '{file_path}' does not exist."
            data_tools_logger.error(f"FILE_NOT_FOUND: {file_path}")
            return error_msg
        
        # Determine file type and read accordingly
        if file_path.suffix.lower() in ['.csv']:
            # Read CSV with smart header detection
            df = pd.read_csv(file_path)
            
            # Check if first column is empty or has mostly empty values
            first_col_filled = df.iloc[:, 0].notna().sum()
            total_rows = len(df)
            
            # If first column is mostly empty, try to find a better header row
            if first_col_filled < total_rows * 0.1:  # Less than 10% filled
                data_tools_logger.info(f"FIRST_COLUMN_EMPTY: {file_path} - First column has {first_col_filled}/{total_rows} filled values")
                
                # Find the first row with many filled fields
                filled_counts = df.notna().sum(axis=1)
                best_header_row = filled_counts.idxmax()
                
                if filled_counts[best_header_row] > 3:  # At least 3 filled fields
                    data_tools_logger.info(f"USING_ROW_AS_HEADER: {file_path} - Row {best_header_row} has {filled_counts[best_header_row]} filled fields")
                    
                    # Read CSV again using the identified header row
                    df = pd.read_csv(file_path, header=best_header_row)
                    
                    # If we used a row as header, we might need to skip some rows
                    if best_header_row > 0:
                        # Skip rows before the header
                        df = pd.read_csv(file_path, header=best_header_row, skiprows=range(best_header_row))
                else:
                    data_tools_logger.warning(f"NO_GOOD_HEADER_ROW: {file_path} - Best row has only {filled_counts[best_header_row]} filled fields")
            
            data_tools_logger.info(f"FILE_READ: {file_path} (CSV) - Shape: {df.shape} - Columns: {list(df.columns)}")
            
        elif file_path.suffix.lower() in ['.xlsx', '.xls']:
            # For Excel files, try to find the best sheet and header row
            try:
                # First try reading with default settings
                df = pd.read_excel(file_path)
                
                # Check if first column is mostly empty
                first_col_filled = df.iloc[:, 0].notna().sum()
                total_rows = len(df)
                
                if first_col_filled < total_rows * 0.1:  # Less than 10% filled
                    data_tools_logger.info(f"FIRST_COLUMN_EMPTY_EXCEL: {file_path} - First column has {first_col_filled}/{total_rows} filled values")
                    
                    # Try reading with header=None to find the best header row
                    df_no_header = pd.read_excel(file_path, header=None)
                    filled_counts = df_no_header.notna().sum(axis=1)
                    best_header_row = filled_counts.idxmax()
                    
                    if filled_counts[best_header_row] > 3:  # At least 3 filled fields
                        data_tools_logger.info(f"USING_ROW_AS_HEADER_EXCEL: {file_path} - Row {best_header_row} has {filled_counts[best_header_row]} filled fields")
                        df = pd.read_excel(file_path, header=best_header_row)
                    else:
                        data_tools_logger.warning(f"NO_GOOD_HEADER_ROW_EXCEL: {file_path} - Best row has only {filled_counts[best_header_row]} filled fields")
                
                data_tools_logger.info(f"FILE_READ: {file_path} (Excel) - Shape: {df.shape} - Columns: {list(df.columns)}")
                
            except Exception as e:
                data_tools_logger.error(f"EXCEL_READ_ERROR: {file_path} - Error: {str(e)}")
                # Try reading without header as fallback
                df = pd.read_excel(file_path, header=None)
                data_tools_logger.info(f"FILE_READ_FALLBACK: {file_path} (Excel) - Shape: {df.shape} - No header used")
        else:
            error_msg = f"Error: Unsupported file format '{file_path.suffix}'. Supported formats: .csv, .xlsx, .xls"
            data_tools_logger.error(f"UNSUPPORTED_FORMAT: {file_path.suffix}")
            return error_msg
        
        # Perform the requested operation
        result = await _execute_data_operation(df, operation, operation_params or {})
        
        # Log successful operation
        data_tools_logger.info(f"OPERATION_SUCCESS: {operation} on {file_path} - Result length: {len(str(result))}")
        return result
        
    except Exception as e:
        error_msg = f"Error performing operation '{operation}' on file '{file_path}': {str(e)}"
        data_tools_logger.error(f"OPERATION_ERROR: {operation} on {file_path} - Error: {str(e)}")
        return error_msg


async def _execute_data_operation(df: pd.DataFrame, operation: str, params: Dict[str, Any]) -> str:
    """Execute the specified data operation on the DataFrame."""
    
    try:
        if operation == "read_data":
            return _format_dataframe_info(df, "Data Overview")
            
        elif operation == "info":
            buffer = io.StringIO()
            df.info(buf=buffer, max_cols=None, memory_usage=True)
            return f"DataFrame Information:\n{buffer.getvalue()}"
            
        elif operation == "head":
            n = params.get("n", 5)
            return f"First {n} rows:\n{df.head(n).to_string()}"
            
        elif operation == "tail":
            n = params.get("n", 5)
            return f"Last {n} rows:\n{df.tail(n).to_string()}"
            
        elif operation == "describe":
            return f"Statistical Summary:\n{df.describe().to_string()}"
            
        elif operation == "columns":
            return f"Columns: {list(df.columns)}"
            
        elif operation == "shape":
            return f"DataFrame shape: {df.shape} (rows, columns)"
            
        elif operation == "filter":
            condition = params.get("condition")
            if not condition:
                return "Error: 'condition' parameter is required for filter operation"
            
            try:
                # Evaluate the condition safely
                filtered_df = df.query(condition)
                return f"Filtered data (condition: {condition}):\n{filtered_df.to_string()}\n\nShape: {filtered_df.shape}"
            except Exception as e:
                return f"Error evaluating filter condition '{condition}': {str(e)}"
                
        elif operation == "sort":
            by = params.get("by", df.columns[0])
            ascending = params.get("ascending", True)
            sorted_df = df.sort_values(by=by, ascending=ascending)
            return f"Sorted data (by: {by}, ascending: {ascending}):\n{sorted_df.head(10).to_string()}"
            
        elif operation == "groupby":
            group_cols = params.get("group_cols", [])
            agg_cols = params.get("agg_cols", [])
            agg_funcs = params.get("agg_funcs", ["mean"])
            
            if not group_cols:
                return "Error: 'group_cols' parameter is required for groupby operation"
            
            grouped = df.groupby(group_cols)
            if agg_cols and agg_funcs:
                result = grouped[agg_cols].agg(agg_funcs)
            else:
                result = grouped.size()
            
            return f"Grouped data:\n{result.to_string()}"
            
        elif operation == "aggregate":
            agg_cols = params.get("agg_cols", df.select_dtypes(include=[np.number]).columns.tolist())
            agg_funcs = params.get("agg_funcs", ["mean", "sum", "count", "min", "max"])
            
            if not agg_cols:
                return "Error: No numeric columns found for aggregation"
            
            result = df[agg_cols].agg(agg_funcs)
            return f"Aggregation results:\n{result.to_string()}"
            
        elif operation == "merge":
            other_file = params.get("other_file")
            on = params.get("on")
            how = params.get("how", "inner")
            
            if not other_file or not on:
                return "Error: 'other_file' and 'on' parameters are required for merge operation"
            
            # Read the other file
            other_path = Path(other_file)
            if other_path.suffix.lower() in ['.csv']:
                other_df = pd.read_csv(other_path)
            elif other_path.suffix.lower() in ['.xlsx', '.xls']:
                other_df = pd.read_excel(other_path)
            else:
                return f"Error: Unsupported file format for merge: {other_path.suffix}"
            
            merged_df = pd.merge(df, other_df, on=on, how=how)
            return f"Merged data ({how} join on {on}):\n{merged_df.head(10).to_string()}\n\nShape: {merged_df.shape}"
            
        elif operation == "pivot":
            index = params.get("index")
            columns = params.get("columns")
            values = params.get("values")
            
            if not all([index, columns, values]):
                return "Error: 'index', 'columns', and 'values' parameters are required for pivot operation"
            
            pivot_df = df.pivot_table(index=index, columns=columns, values=values, aggfunc='mean')
            return f"Pivot table:\n{pivot_df.to_string()}"
            
        elif operation == "correlation":
            numeric_df = df.select_dtypes(include=[np.number])
            if numeric_df.empty:
                return "Error: No numeric columns found for correlation analysis"
            
            corr_matrix = numeric_df.corr()
            return f"Correlation matrix:\n{corr_matrix.to_string()}"
            
        elif operation == "missing_values":
            missing_info = df.isnull().sum()
            missing_percent = (missing_info / len(df)) * 100
            missing_df = pd.DataFrame({
                'Missing_Count': missing_info,
                'Missing_Percent': missing_percent
            })
            return f"Missing values analysis:\n{missing_df.to_string()}"
            
        elif operation == "duplicates":
            duplicate_count = df.duplicated().sum()
            duplicate_rows = df[df.duplicated()]
            return f"Duplicate analysis:\nTotal duplicates: {duplicate_count}\n\nDuplicate rows:\n{duplicate_rows.to_string()}"
            
        elif operation == "value_counts":
            column = params.get("column")
            if not column:
                return "Error: 'column' parameter is required for value_counts operation"
            
            if column not in df.columns:
                return f"Error: Column '{column}' not found in DataFrame"
            
            value_counts = df[column].value_counts()
            return f"Value counts for column '{column}':\n{value_counts.to_string()}"
            
        elif operation == "sample":
            n = params.get("n", 10)
            random_state = params.get("random_state", None)
            sampled_df = df.sample(n=n, random_state=random_state)
            return f"Random sample ({n} rows):\n{sampled_df.to_string()}"
            
        elif operation == "drop_columns":
            columns_to_drop = params.get("columns", [])
            if not columns_to_drop:
                return "Error: 'columns' parameter is required for drop_columns operation"
            
            df_dropped = df.drop(columns=columns_to_drop)
            return f"Columns dropped: {columns_to_drop}\nNew shape: {df_dropped.shape}\n\nFirst few rows:\n{df_dropped.head().to_string()}"
            
        elif operation == "rename_columns":
            rename_map = params.get("rename_map", {})
            if not rename_map:
                return "Error: 'rename_map' parameter is required for rename_columns operation"
            
            df_renamed = df.rename(columns=rename_map)
            return f"Columns renamed:\n{rename_map}\n\nNew columns: {list(df_renamed.columns)}\n\nFirst few rows:\n{df_renamed.head().to_string()}"
            
        elif operation == "fill_missing":
            method = params.get("method", "ffill")  # forward fill
            value = params.get("value")
            
            if value is not None:
                df_filled = df.fillna(value)
            else:
                df_filled = df.fillna(method=method)
            
            return f"Missing values filled (method: {method}):\n{df_filled.head().to_string()}"
            
        elif operation == "drop_missing":
            axis = params.get("axis", 0)  # 0 for rows, 1 for columns
            how = params.get("how", "any")  # 'any' or 'all'
            
            df_cleaned = df.dropna(axis=axis, how=how)
            return f"Missing values dropped (axis: {axis}, how: {how}):\nShape: {df_cleaned.shape}\n\nFirst few rows:\n{df_cleaned.head().to_string()}"
            
        elif operation == "convert_types":
            type_conversions = params.get("type_conversions", {})
            if not type_conversions:
                return "Error: 'type_conversions' parameter is required for convert_types operation"
            
            df_converted = df.copy()
            for col, dtype in type_conversions.items():
                if col in df_converted.columns:
                    try:
                        df_converted[col] = df_converted[col].astype(dtype)
                    except Exception as e:
                        return f"Error converting column '{col}' to {dtype}: {str(e)}"
            
            return f"Type conversions applied:\n{type_conversions}\n\nData types:\n{df_converted.dtypes.to_string()}"
            
        elif operation == "search":
            search_term = params.get("search_term")
            column = params.get("column")
            case_sensitive = params.get("case_sensitive", False)
            partial_match = params.get("partial_match", True)
            
            # Log search operation details
            search_log_data = {
                "operation": "search",
                "search_term": search_term,
                "column": column,
                "case_sensitive": case_sensitive,
                "partial_match": partial_match,
                "dataframe_shape": df.shape,
                "available_columns": list(df.columns)
            }
            data_tools_logger.info(f"SEARCH_OPERATION: {json.dumps(search_log_data, indent=2)}")
            
            if not search_term:
                data_tools_logger.error("SEARCH_ERROR: Missing search_term parameter")
                return "Error: 'search_term' parameter is required for search operation"
            
            try:
                # If column is specified, search only in that column
                if column:
                    if column not in df.columns:
                        return f"Error: Column '{column}' not found in DataFrame. Available columns: {list(df.columns)}"
                    
                    # Search in specific column
                    if case_sensitive:
                        if partial_match:
                            mask = df[column].astype(str).str.contains(search_term, na=False)
                        else:
                            mask = df[column].astype(str) == search_term
                    else:
                        if partial_match:
                            mask = df[column].astype(str).str.contains(search_term, case=False, na=False)
                        else:
                            mask = df[column].astype(str).str.lower() == search_term.lower()
                else:
                    # Search across all columns
                    mask = pd.Series([False] * len(df))
                    for col in df.columns:
                        if case_sensitive:
                            if partial_match:
                                col_mask = df[col].astype(str).str.contains(search_term, na=False)
                            else:
                                col_mask = df[col].astype(str) == search_term
                        else:
                            if partial_match:
                                col_mask = df[col].astype(str).str.contains(search_term, case=False, na=False)
                            else:
                                col_mask = df[col].astype(str).str.lower() == search_term.lower()
                        mask = mask | col_mask
                
                # Filter the DataFrame
                search_results = df[mask]
                
                # Log search results
                data_tools_logger.info(f"SEARCH_RESULTS: '{search_term}' - Found {len(search_results)} matches out of {len(df)} total rows")
                
                # Generate search summary
                total_matches = len(search_results)
                search_summary = f"Search Results for '{search_term}':\n"
                search_summary += f"Total matches found: {total_matches}\n"
                
                if column:
                    search_summary += f"Searching in column: '{column}'\n"
                else:
                    search_summary += "Searching across all columns\n"
                
                search_summary += f"Case sensitive: {case_sensitive}\n"
                search_summary += f"Partial match: {partial_match}\n"
                search_summary += f"Total rows in dataset: {len(df)}\n"
                search_summary += f"Match percentage: {(total_matches/len(df)*100):.1f}%\n\n"
                
                if total_matches == 0:
                    search_summary += "No matches found."
                else:
                    # Show all results (or limit if too many)
                    max_display = params.get("max_display", 50)
                    if total_matches > max_display:
                        search_summary += f"Showing first {max_display} results (out of {total_matches}):\n\n"
                        display_results = search_results.head(max_display)
                    else:
                        search_summary += "All matching results:\n\n"
                        display_results = search_results
                    
                    search_summary += display_results.to_string(index=False)
                    
                    if total_matches > max_display:
                        search_summary += f"\n\n... and {total_matches - max_display} more results"
                
                return search_summary
                
            except Exception as e:
                return f"Error performing search: {str(e)}"
            
        elif operation == "save":
            output_path = params.get("output_path")
            format_type = params.get("format", "csv")
            
            if not output_path:
                return "Error: 'output_path' parameter is required for save operation"
            
            output_path = Path(output_path)
            if format_type.lower() == "csv":
                df.to_csv(output_path, index=False)
            elif format_type.lower() in ["excel", "xlsx"]:
                df.to_excel(output_path, index=False)
            else:
                return f"Error: Unsupported output format '{format_type}'. Supported: csv, excel, xlsx"
            
            return f"Data saved to: {output_path}"
            
        else:
            return f"Error: Unknown operation '{operation}'"
            
    except Exception as e:
        return f"Error executing operation '{operation}': {str(e)}"


def _format_dataframe_info(df: pd.DataFrame, title: str) -> str:
    """Format DataFrame information in a readable way."""
    info = f"{title}\n"
    info += f"Shape: {df.shape} (rows, columns)\n"
    info += f"Columns: {list(df.columns)}\n"
    info += f"Data types:\n{df.dtypes.to_string()}\n"
    info += f"First 5 rows:\n{df.head().to_string()}\n"
    
    # Add basic statistics for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        info += f"\nNumeric columns summary:\n{df[numeric_cols].describe().to_string()}"
    
    return info


##########################
# List Data Files Tool
##########################
LIST_DATA_FILES_DESCRIPTION = (
    "A tool for exploring and listing all files in a data folder. Use this to discover available datasets before using the CSV/Excel analysis tool. Supports filtering by file type and provides file information."
)

@tool(description=LIST_DATA_FILES_DESCRIPTION)
async def list_data_files(
    folder_path: str = "data",
    file_types: Optional[List[str]] = None,
    include_subfolders: bool = False,
    show_details: bool = True,
    config: RunnableConfig = None
) -> str:
    """
    List all files in a specified data folder with optional filtering and detailed information.
    
    Args:
        folder_path (str): Path to the folder to list files from (default: "data")
        file_types (List[str]): Optional list of file extensions to filter by (e.g., [".csv", ".xlsx"])
        include_subfolders (bool): Whether to include files from subfolders (default: False)
        show_details (bool): Whether to show file details like size and modification date (default: True)
        config (RunnableConfig): Configuration object (optional)
    
    Returns:
        str: Formatted list of files with details
    """
    # Log the tool call
    log_data = {
        "tool": "list_data_files",
        "folder_path": folder_path,
        "file_types": file_types,
        "include_subfolders": include_subfolders,
        "show_details": show_details,
        "timestamp": datetime.now().isoformat(),
        "config_keys": list(config.keys()) if config else []
    }
    data_tools_logger.info(f"TOOL_CALL: {json.dumps(log_data, indent=2)}")
    
    try:
        folder_path = Path(folder_path)
        
        # Check if folder exists
        if not folder_path.exists():
            error_msg = f"Error: Folder '{folder_path}' does not exist."
            data_tools_logger.error(f"FOLDER_NOT_FOUND: {folder_path}")
            return error_msg
        
        if not folder_path.is_dir():
            error_msg = f"Error: '{folder_path}' is not a directory."
            data_tools_logger.error(f"NOT_A_DIRECTORY: {folder_path}")
            return error_msg
        
        # Define supported file types for data analysis
        supported_types = {
            '.csv': 'CSV Data File',
            '.xlsx': 'Excel Spreadsheet',
            '.xls': 'Excel Spreadsheet (Legacy)',
            '.json': 'JSON Data File',
            '.parquet': 'Parquet Data File',
            '.feather': 'Feather Data File',
            '.h5': 'HDF5 Data File',
            '.hdf5': 'HDF5 Data File',
            '.pkl': 'Pickle Data File',
            '.pickle': 'Pickle Data File',
            '.txt': 'Text Data File',
            '.tsv': 'Tab-Separated Values',
            '.dat': 'Data File',
            '.db': 'Database File',
            '.sqlite': 'SQLite Database',
            '.sql': 'SQL Script'
        }
        
        # Collect files
        files = []
        if include_subfolders:
            for file_path in folder_path.rglob("*"):
                if file_path.is_file():
                    files.append(file_path)
        else:
            for file_path in folder_path.iterdir():
                if file_path.is_file():
                    files.append(file_path)
        
        # Filter by file type if specified
        if file_types:
            filtered_files = []
            for file_path in files:
                if file_path.suffix.lower() in [ext.lower() for ext in file_types]:
                    filtered_files.append(file_path)
            files = filtered_files
        
        if not files:
            if file_types:
                return f"No files found in '{folder_path}' with specified types: {file_types}"
            else:
                return f"No files found in '{folder_path}'"
        
        # Sort files by name
        files.sort(key=lambda x: x.name.lower())
        
        # Generate report
        report = f"Files found in '{folder_path}':\n"
        report += f"Total files: {len(files)}\n"
        
        if file_types:
            report += f"Filtered by types: {file_types}\n"
        
        if include_subfolders:
            report += "Including subfolders: Yes\n"
        
        report += "\n" + "="*80 + "\n\n"
        
        # Group files by type
        files_by_type = {}
        for file_path in files:
            file_type = file_path.suffix.lower()
            if file_type not in files_by_type:
                files_by_type[file_type] = []
            files_by_type[file_type].append(file_path)
        
        # Sort file types
        sorted_types = sorted(files_by_type.keys())
        
        for file_type in sorted_types:
            type_files = files_by_type[file_type]
            type_name = supported_types.get(file_type, f"{file_type.upper()} File")
            
            report += f"## {type_name} ({len(type_files)} files)\n\n"
            
            for file_path in type_files:
                report += f"**{file_path.name}**\n"
                
                if show_details:
                    try:
                        # Get file stats
                        stat = file_path.stat()
                        size_bytes = stat.st_size
                        modified_time = datetime.fromtimestamp(stat.st_mtime)
                        
                        # Format file size
                        if size_bytes < 1024:
                            size_str = f"{size_bytes} B"
                        elif size_bytes < 1024**2:
                            size_str = f"{size_bytes/1024:.1f} KB"
                        elif size_bytes < 1024**3:
                            size_str = f"{size_bytes/1024**2:.1f} MB"
                        else:
                            size_str = f"{size_bytes/1024**3:.1f} GB"
                        
                        # Get relative path
                        rel_path = file_path.relative_to(folder_path)
                        
                        report += f"  - Path: {rel_path}\n"
                        report += f"  - Size: {size_str}\n"
                        report += f"  - Modified: {modified_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                        
                        # Add preview for supported file types
                        if file_type in ['.csv', '.xlsx', '.xls']:
                            try:
                                if file_type == '.csv':
                                    df = pd.read_csv(file_path, nrows=3)
                                else:
                                    df = pd.read_excel(file_path, nrows=3)
                                
                                report += f"  - Preview: {df.shape[0]} rows, {df.shape[1]} columns\n"
                                report += f"  - Columns: {list(df.columns)[:5]}{'...' if len(df.columns) > 5 else ''}\n"
                            except Exception as e:
                                report += f"  - Preview: Unable to read file ({str(e)[:50]}...)\n"
                        
                    except Exception as e:
                        report += f"  - Error getting file details: {str(e)}\n"
                
                report += "\n"
        
        # Add summary statistics
        report += "## Summary\n\n"
        report += f"- Total files: {len(files)}\n"
        report += f"- File types found: {len(files_by_type)}\n"
        
        # File type breakdown
        type_breakdown = []
        for file_type, type_files in files_by_type.items():
            type_name = supported_types.get(file_type, file_type.upper())
            type_breakdown.append(f"{type_name}: {len(type_files)}")
        
        report += f"- File type breakdown: {', '.join(type_breakdown)}\n"
        
        # Total size
        total_size = 0
        for file_path in files:
            try:
                total_size += file_path.stat().st_size
            except:
                pass
        
        if total_size > 0:
            if total_size < 1024**2:
                total_size_str = f"{total_size/1024:.1f} KB"
            elif total_size < 1024**3:
                total_size_str = f"{total_size/1024**2:.1f} MB"
            else:
                total_size_str = f"{total_size/1024**3:.1f} GB"
            report += f"- Total size: {total_size_str}\n"
        
        # Log successful operation
        data_tools_logger.info(f"LIST_FILES_SUCCESS: {folder_path} - Found {len(files)} files")
        return report
        
    except Exception as e:
        error_msg = f"Error listing files in '{folder_path}': {str(e)}"
        data_tools_logger.error(f"LIST_FILES_ERROR: {folder_path} - Error: {str(e)}")
        return error_msg