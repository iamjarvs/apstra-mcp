#!/usr/bin/env python3
"""
MCP Server with OpenAPI Integration using FastMCP
Automatically generates MCP functions from OpenAPI specifications
"""

import asyncio
import json
import logging
import os
import sys
from typing import Any, Dict, Optional
from urllib.parse import urljoin

import httpx
import yaml
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel

# Try to load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Response templates for Claude
RESPONSE_TEMPLATES = {
    'blueprint_summary': """
## Blueprint Summary: {name}
- **Status**: {status}
- **Design**: {design_type} 
- **Infrastructure**: {leaf_count} leaves, {spine_count} spines
- **Anomalies**: {anomaly_count} detected
- **Security Zones**: {vrf_count}

{additional_details}
""",
    
    'node_summary': """
## Network Nodes Summary
**Total Nodes**: {total_count}

### By Type:
{node_type_breakdown}

### Key Infrastructure:
{infrastructure_table}
""",
    
    'error_summary': """
## Network Validation Errors
**Total Errors**: {error_count}
**Severity Breakdown**: Critical: {critical}, Warning: {warning}, Info: {info}

### Critical Issues:
{critical_errors}

### Recommendations:
{recommendations}
"""
}


class AuthConfig(BaseModel):
    username: str
    password: str
    base_url: str
    token: Optional[str] = None
    token_header: str = "AUTHTOKEN"  # Default header name for token


class OpenAPIFunction:
    """Represents a function derived from an OpenAPI operation"""
    
    def __init__(self, path: str, method: str, operation: Dict[str, Any], base_url: str):
        self.path = path
        self.method = method.upper()
        self.operation = operation
        self.base_url = base_url
        self.name = self._generate_function_name()
        self.description = self._extract_description()
    
    def _generate_function_name(self) -> str:
        """Generate a function name from the operation"""
        # Use operationId if available, otherwise generate from path and method
        if 'operationId' in self.operation:
            return self.operation['operationId']
        
        # Clean up path and combine with method
        clean_path = self.path.replace('/', '_').replace('{', '').replace('}', '').strip('_')
        return f"{self.method.lower()}_{clean_path}"
    
    def _extract_description(self) -> str:
        """Extract description for Claude with formatting guidance"""
        summary = self.operation.get('summary', '')
        description = self.operation.get('description', '')
        
        base_description = summary or description or f"{self.method} {self.path}"
        
        # Add formatting guidance based on the endpoint
        formatting_guidance = ""
        
        if 'blueprints' in self.path and self.method == 'GET':
            if '{blueprint_id}' not in self.path:
                # List all blueprints
                formatting_guidance = "\n\nWhen presenting results, format as a table showing: Blueprint ID, Design Type, Status, Anomaly Count, and brief description of each blueprint's purpose."
            elif '/nodes' in self.path and '{node_id}' not in self.path:
                # List nodes in blueprint
                formatting_guidance = "\n\nWhen presenting results, summarise the node types and counts first, then provide a table of key nodes with their IDs, types, and important properties. Group similar node types together for clarity."
            elif '/nodes/{node_id}' in self.path:
                # Get specific node
                formatting_guidance = "\n\nWhen presenting results, extract and highlight the key properties of this node including its type, role, configuration details, and any operational status. Present technical details in a structured, readable format."
            elif '/errors' in self.path:
                # Get errors
                formatting_guidance = "\n\nWhen presenting results, prioritise and categorise errors by severity. Provide clear summaries of what each error means and potential impact on network operations."
            else:
                # Get specific blueprint
                formatting_guidance = "\n\nWhen presenting results, provide a high-level summary of the blueprint's design and operational state first, then drill down into specific components as requested. Focus on anomalies and operational issues."
        
        return f"{base_description}{formatting_guidance}"
    
    async def execute(self, auth: AuthConfig, **kwargs) -> Dict[str, Any]:
        """Execute the API call"""
        url = urljoin(auth.base_url, self.path)
        
        # Replace path parameters
        for key, value in kwargs.items():
            if f"{{{key}}}" in url:
                url = url.replace(f"{{{key}}}", str(value))
        
        # Prepare request data
        params = {}
        json_data = {}
        
        # Separate path/query params from body data
        path_params = []
        query_params = []
        
        for param in self.operation.get('parameters', []):
            if param.get('in') == 'path':
                path_params.append(param['name'])
            elif param.get('in') == 'query':
                query_params.append(param['name'])
        
        # Build query parameters and request body
        for key, value in kwargs.items():
            if key in query_params:
                params[key] = value
            elif key not in path_params:
                json_data[key] = value
        
        # Prepare headers with authentication
        headers = {}
        if auth.token:
            headers[auth.token_header] = auth.token
        
        # Get SSL verification setting
        verify_ssl = os.getenv('MCP_VERIFY_SSL', 'true').lower() in ('true', '1', 'yes', 'on')
        
        # Make the HTTP request
        async with httpx.AsyncClient(verify=verify_ssl) as client:
            try:
                # Use token authentication if available, otherwise fall back to basic auth
                auth_method = None if auth.token else (auth.username, auth.password)
                
                response = await client.request(
                    method=self.method,
                    url=url,
                    params=params if params else None,
                    json=json_data if json_data else None,
                    headers=headers,
                    auth=auth_method,
                    timeout=30.0
                )
                response.raise_for_status()
                
                # Try to parse as JSON, fall back to text
                try:
                    result = response.json()
                    return result
                except json.JSONDecodeError:
                    return {"response": response.text, "status_code": response.status_code}
                    
            except httpx.HTTPError as e:
                logger.error(f"HTTP error calling {url}: {e}")
                return {"error": str(e), "status_code": getattr(e.response, 'status_code', None)}
    
    def _get_formatting_hint(self) -> str:
        """Get formatting hint for Claude based on the endpoint"""
        if 'blueprints' in self.path and self.method == 'GET':
            if '{blueprint_id}' not in self.path:
                return "Format as table: ID | Name | Design Type | Status | Anomaly Count | Description"
            elif '/nodes' in self.path and '{node_id}' not in self.path:
                return "Summarise node types and counts first, then show key nodes in table format grouped by type"
            elif '/nodes/{node_id}' in self.path:
                return "Extract key properties: type, role, configuration, status. Present in structured, readable format"
            elif '/errors' in self.path:
                return "Categorise by severity, provide clear error descriptions and potential impact"
            else:
                return "Provide high-level summary first, then specific details. Highlight any anomalies or issues"
        return "Present data in clear, structured format appropriate for network operations"
    
    def _post_process_response(self, data: Any) -> Dict[str, Any]:
        """Post-process API responses to make them more Claude-friendly"""
        
        endpoint_type = self._get_endpoint_type()
        
        if endpoint_type == "blueprint_list":
            # For /api/blueprints - extract key info
            if isinstance(data, dict) and 'items' in data:
                processed = []
                for blueprint in data['items']:
                    processed.append({
                        'id': blueprint.get('id'),
                        'name': blueprint.get('label', 'Unnamed'),
                        'design_type': blueprint.get('design', {}).get('type'),
                        'status': blueprint.get('status'),
                        'anomaly_count': sum(blueprint.get('anomaly_counts', {}).values()) if blueprint.get('anomaly_counts') else 0,
                        'summary': f"Blueprint with {blueprint.get('design', {}).get('leaf_count', 0)} leaves, {blueprint.get('design', {}).get('spine_count', 0)} spines"
                    })
                return {
                    'blueprints': processed,
                    '_count': len(processed),
                    '_formatting_hint': "Show as table with ID, Name, Type, Status, and Anomaly Count columns",
                    '_template': RESPONSE_TEMPLATES.get('blueprint_summary', '')
                }
        
        elif endpoint_type == "blueprint_nodes":
            # For /api/blueprints/{id}/nodes - summarise node types
            if isinstance(data, dict) and 'nodes' in data:
                node_summary = {}
                important_nodes = []
                
                for node_id, node in data['nodes'].items():
                    node_type = node.get('type', 'unknown')
                    node_summary[node_type] = node_summary.get(node_type, 0) + 1
                    
                    # Extract important nodes (switches, routers, etc.)
                    if node_type in ['switch', 'router', 'spine', 'leaf', 'system']:
                        important_nodes.append({
                            'id': node_id,
                            'type': node_type,
                            'label': node.get('label', 'Unnamed'),
                            'role': node.get('role'),
                            'status': node.get('status', 'unknown')
                        })
                
                return {
                    'node_summary': node_summary,
                    'important_nodes': important_nodes[:20],  # Limit to first 20
                    'total_nodes': len(data['nodes']),
                    'raw_data': data,  # Keep original data available
                    '_formatting_hint': "Show node type summary first, then table of important infrastructure nodes",
                    '_template': RESPONSE_TEMPLATES.get('node_summary', '')
                }
        
        elif endpoint_type == "blueprint_errors":
            # For /api/blueprints/{id}/errors - categorise errors
            if isinstance(data, dict):
                error_count = 0
                critical_errors = []
                
                # Count and categorise errors (this depends on Apstra's error structure)
                for error_type, errors in data.items():
                    if isinstance(errors, list):
                        error_count += len(errors)
                        # Add severity logic based on error type
                        for error in errors[:5]:  # Show first 5 of each type
                            critical_errors.append({
                                'type': error_type,
                                'message': str(error),
                                'severity': 'critical' if 'critical' in error_type.lower() else 'warning'
                            })
                
                return {
                    'error_summary': {
                        'total_count': error_count,
                        'critical_errors': critical_errors,
                    },
                    'raw_data': data,
                    '_formatting_hint': "Categorise by severity, provide clear error descriptions and potential impact",
                    '_template': RESPONSE_TEMPLATES.get('error_summary', '')
                }
        
        # Default: return original data with basic hint
        return {
            'data': data,
            '_formatting_hint': self._get_formatting_hint()
        }
    
    def _get_endpoint_type(self) -> str:
        """Determine the type of endpoint for processing"""
        if 'blueprints' in self.path:
            if '{blueprint_id}' not in self.path:
                return "blueprint_list"
            elif '/nodes' in self.path and '{node_id}' not in self.path:
                return "blueprint_nodes"
            elif '/nodes/{node_id}' in self.path:
                return "blueprint_node_detail"
            elif '/errors' in self.path:
                return "blueprint_errors"
            else:
                return "blueprint_detail"
        return "generic"
    
    def _add_schema_hints(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Add schema hints to help Claude understand complex nested data"""
        
        # Add hints based on data structure
        if isinstance(data, dict):
            if 'nodes' in data or (isinstance(data.get('data'), dict) and 'nodes' in data['data']):
                data['_schema_hint'] = {
                    'nodes': 'Dictionary of node_id -> node_properties',
                    'key_fields': ['type', 'label', 'role', 'status', 'properties'],
                    'common_types': ['switch', 'spine', 'leaf', 'server', 'external_router', 'system'],
                    'tip': 'Focus on nodes with type switch, spine, leaf for network infrastructure'
                }
            
            if 'relationships' in data or (isinstance(data.get('data'), dict) and 'relationships' in data['data']):
                data['_schema_hint'] = {
                    'relationships': 'Dictionary of relationship_id -> relationship_properties', 
                    'key_fields': ['type', 'source', 'target', 'properties'],
                    'common_types': ['spine_leaf', 'to_generic', 'member_interfaces'],
                    'tip': 'Relationships show how network components connect to each other'
                }
            
            # Add general Apstra schema context
            if not data.get('_schema_hint'):
                data['_schema_hint'] = {
                    'context': 'Apstra network management data',
                    'tip': 'Look for id, label, type, and status fields for key information'
                }
        
        return data


class OpenAPIMCPServer:
    """MCP Server that dynamically creates tools from OpenAPI specs"""
    
    def __init__(self):
        self.mcp = FastMCP("openapi-mcp-server")
        self.functions: Dict[str, OpenAPIFunction] = {}
        self.auth_config: Optional[AuthConfig] = None
        self._auth_initialized = False
    
    def load_auth_config(self) -> AuthConfig:
        """Load authentication configuration from environment variables"""
        username = os.getenv('MCP_USERNAME')
        password = os.getenv('MCP_PASSWORD')
        base_url = os.getenv('MCP_BASE_URL')
        login_endpoint = os.getenv('MCP_LOGIN_ENDPOINT', '/api/user/login')  # Default login endpoint
        token_header = os.getenv('MCP_TOKEN_HEADER', 'AUTHTOKEN')  # Default token header name for Apstra
        verify_ssl = os.getenv('MCP_VERIFY_SSL', 'true').lower() in ('true', '1', 'yes', 'on')
        
        # Debug: print what we found
        logger.info(f"Environment variables - Username: {'***' if username else 'None'}, "
                   f"Password: {'***' if password else 'None'}, "
                   f"Base URL: {base_url or 'None'}, "
                   f"Login endpoint: {login_endpoint}, "
                   f"Token header: {token_header}, "
                   f"Verify SSL: {verify_ssl}")
        
        if not verify_ssl:
            logger.warning("SSL verification is disabled - this should only be used in development!")
        
        if not all([username, password, base_url]):
            # Try to help with troubleshooting
            missing = []
            if not username: missing.append('MCP_USERNAME')
            if not password: missing.append('MCP_PASSWORD') 
            if not base_url: missing.append('MCP_BASE_URL')
            
            error_msg = f"Missing required environment variables: {', '.join(missing)}\n"
            error_msg += "Please ensure your .env file is in the current directory with:\n"
            error_msg += "MCP_USERNAME=your_username\n"
            error_msg += "MCP_PASSWORD=your_password\n"
            error_msg += "MCP_BASE_URL=https://your-api-base-url.com\n"
            error_msg += "MCP_LOGIN_ENDPOINT=/api/user/login  # Optional, defaults to /api/user/login\n"
            error_msg += "MCP_TOKEN_HEADER=AUTHTOKEN  # Optional, defaults to AUTHTOKEN\n"
            error_msg += "MCP_VERIFY_SSL=false  # Optional, set to false for self-signed certificates"
            
            raise ValueError(error_msg)
        
        return AuthConfig(
            username=username, 
            password=password, 
            base_url=base_url,
            token_header=token_header
        )
    
    async def authenticate_and_get_token(self) -> str:
        """Authenticate with the API and get a token"""
        if not self.auth_config:
            raise ValueError("Auth config not loaded")
        
        login_endpoint = os.getenv('MCP_LOGIN_ENDPOINT', '/api/user/login')
        login_url = urljoin(self.auth_config.base_url, login_endpoint)
        verify_ssl = os.getenv('MCP_VERIFY_SSL', 'true').lower() in ('true', '1', 'yes', 'on')
        
        # Prepare login payload
        login_payload = {
            'username': self.auth_config.username,
            'password': self.auth_config.password
        }
        
        logger.info(f"Attempting to authenticate at {login_url}")
        
        async with httpx.AsyncClient(verify=verify_ssl) as client:
            try:
                response = await client.post(
                    login_url,
                    json=login_payload,
                    timeout=30.0
                )
                response.raise_for_status()
                
                # Parse the response to extract the token
                try:
                    auth_response = response.json()
                    
                    # Try common token field names
                    token = (auth_response.get('token') or 
                            auth_response.get('access_token') or 
                            auth_response.get('authToken') or
                            auth_response.get('auth_token'))
                    
                    if not token:
                        # If no standard field found, log the response structure
                        logger.error(f"Token not found in response. Response keys: {list(auth_response.keys())}")
                        logger.error(f"Full response: {auth_response}")
                        raise ValueError("Authentication succeeded but no token found in response")
                    
                    logger.info("Authentication successful, token obtained")
                    return token
                    
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON response from login endpoint: {response.text}")
                    raise ValueError("Authentication endpoint returned invalid JSON")
                    
            except httpx.HTTPError as e:
                logger.error(f"HTTP error during authentication: {e}")
                if hasattr(e, 'response') and e.response:
                    logger.error(f"Response status: {e.response.status_code}")
                    logger.error(f"Response body: {e.response.text}")
                raise ValueError(f"Authentication failed: {e}")
    
    async def initialize_authentication(self):
        """Initialize authentication by getting a token"""
        try:
            token = await self.authenticate_and_get_token()
            self.auth_config.token = token
            logger.info("Authentication initialization completed successfully")
        except Exception as e:
            logger.error(f"Failed to initialize authentication: {e}")
            raise
    
    async def ensure_authenticated(self):
        """Ensure authentication is initialized before API calls"""
        if not self._auth_initialized:
            await self.initialize_authentication()
            self._auth_initialized = True
    
    def load_openapi_spec(self, spec_path: str):
        """Load and parse OpenAPI specification"""
        try:
            with open(spec_path, 'r') as file:
                if spec_path.endswith('.yaml') or spec_path.endswith('.yml'):
                    spec = yaml.safe_load(file)
                else:
                    spec = json.load(file)
            
            self._parse_openapi_spec(spec)
            logger.info(f"Loaded OpenAPI spec with {len(self.functions)} operations")
            
        except Exception as e:
            logger.error(f"Failed to load OpenAPI spec: {e}")
            raise
    
    def _parse_openapi_spec(self, spec: Dict[str, Any]):
        """Parse OpenAPI spec and create function objects"""
        base_url = self.auth_config.base_url
        paths = spec.get('paths', {})
        
        for path, path_item in paths.items():
            for method, operation in path_item.items():
                if method.lower() in ['get', 'post', 'put', 'patch', 'delete']:
                    func = OpenAPIFunction(path, method, operation, base_url)
                    self.functions[func.name] = func
                    
                    # Register the function with FastMCP
                    self._register_function(func)
                    logger.info(f"Registered function: {func.name}")
    
    def _register_function(self, func: OpenAPIFunction):
        """Register a function with FastMCP"""
        # Create the parameters for the function
        parameters = self._extract_parameters(func.operation, func.method)
        
        # Log what parameters we extracted
        logger.info(f"Function {func.name} parameters: {parameters}")
        
        # Create a wrapper function with explicit parameter names
        # This ensures FastMCP sees the correct parameter signature
        if 'blueprint_id' in parameters and 'node_id' in parameters:
            # Function with both blueprint_id and node_id
            async def api_call(blueprint_id: str, node_id: str, **kwargs):
                logger.info(f"API call {func.name} received blueprint_id={blueprint_id}, node_id={node_id}, kwargs={kwargs}")
                await self.ensure_authenticated()
                return await func.execute(self.auth_config, blueprint_id=blueprint_id, node_id=node_id, **kwargs)
        elif 'blueprint_id' in parameters and 'relationship_id' in parameters:
            # Function with blueprint_id and relationship_id
            async def api_call(blueprint_id: str, relationship_id: str, **kwargs):
                logger.info(f"API call {func.name} received blueprint_id={blueprint_id}, relationship_id={relationship_id}, kwargs={kwargs}")
                await self.ensure_authenticated()
                return await func.execute(self.auth_config, blueprint_id=blueprint_id, relationship_id=relationship_id, **kwargs)
        elif 'blueprint_id' in parameters:
            # Function with just blueprint_id
            if 'node_type' in parameters:
                # Has optional node_type query parameter
                async def api_call(blueprint_id: str, node_type: Optional[str] = None, **kwargs):
                    logger.info(f"API call {func.name} received blueprint_id={blueprint_id}, node_type={node_type}, kwargs={kwargs}")
                    await self.ensure_authenticated()
                    all_params = {'blueprint_id': blueprint_id}
                    if node_type is not None:
                        all_params['node_type'] = node_type
                    all_params.update(kwargs)
                    return await func.execute(self.auth_config, **all_params)
            else:
                # Just blueprint_id
                async def api_call(blueprint_id: str, **kwargs):
                    logger.info(f"API call {func.name} received blueprint_id={blueprint_id}, kwargs={kwargs}")
                    await self.ensure_authenticated()
                    return await func.execute(self.auth_config, blueprint_id=blueprint_id, **kwargs)
        elif 'username' in parameters and 'password' in parameters:
            # Login function
            async def api_call(username: str, password: str, **kwargs):
                logger.info(f"API call {func.name} received username={username}, password=***, kwargs={kwargs}")
                await self.ensure_authenticated()
                return await func.execute(self.auth_config, username=username, password=password, **kwargs)
        else:
            # Generic function with no specific parameters
            async def api_call(**kwargs):
                logger.info(f"API call {func.name} received kwargs: {kwargs}")
                await self.ensure_authenticated()
                return await func.execute(self.auth_config, **kwargs)
        
        # Set the function metadata
        api_call.__name__ = func.name
        api_call.__doc__ = func.description
        
        # Register with FastMCP
        self.mcp.tool()(api_call)
    
    def _extract_parameters(self, operation: Dict[str, Any], method: str) -> Dict[str, Any]:
        """Extract parameters from OpenAPI operation"""
        parameters = {}
        
        # Handle path and query parameters
        for param in operation.get('parameters', []):
            param_name = param['name']
            param_type = param.get('schema', {}).get('type', 'string')
            python_type = self._convert_openapi_type(param_type)
            parameters[param_name] = python_type
            
            # Log parameter details for debugging
            logger.info(f"Parameter: {param_name}, Type: {param_type}, In: {param.get('in')}, Required: {param.get('required', False)}")
        
        # Handle request body for POST/PUT/PATCH
        if method.upper() in ['POST', 'PUT', 'PATCH']:
            request_body = operation.get('requestBody')
            if request_body:
                content = request_body.get('content', {})
                json_content = content.get('application/json', {})
                schema = json_content.get('schema', {})
                
                if schema.get('type') == 'object':
                    schema_properties = schema.get('properties', {})
                    for prop_name, prop_schema in schema_properties.items():
                        param_type = prop_schema.get('type', 'string')
                        python_type = self._convert_openapi_type(param_type)
                        parameters[prop_name] = python_type
                elif not schema:
                    # Handle the case where requestBody has no schema (like in your login endpoints)
                    # Add username and password for login endpoints
                    if 'login' in operation.get('operationId', '').lower():
                        parameters['username'] = str
                        parameters['password'] = str
        
        return parameters
    
    def _convert_openapi_type(self, openapi_type: str):
        """Convert OpenAPI type to Python type"""
        type_mapping = {
            'string': str,
            'integer': int,
            'number': float,
            'boolean': bool,
            'array': list,
            'object': dict
        }
        return type_mapping.get(openapi_type, str)
    
    def _create_annotations(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Create function annotations for type hints"""
        annotations = {}
        for param_name, param_type in parameters.items():
            # Make all parameters optional with default None
            # This allows Claude to call functions with or without all parameters
            annotations[param_name] = Optional[param_type]
        annotations['return'] = Dict[str, Any]
        return annotations
    
    def run(self, openapi_spec_path: str):
        """Start the MCP server"""
        try:
            # Load configuration first
            self.auth_config = self.load_auth_config()
            
            # Load OpenAPI specification and register functions (synchronously)
            self.load_openapi_spec(openapi_spec_path)
            
            # Add smart telemetry functions that handle the async polling
            self._register_smart_telemetry_functions()
            
            # Start the server (authentication will happen on first API call)
            logger.info("Starting OpenAPI MCP Server (authentication will happen on first API call)")
            self.mcp.run()
            
        except Exception as e:
            logger.error(f"Server error: {e}")
            raise
    
    def _register_smart_telemetry_functions(self):
        """Register high-level telemetry functions that handle async operations"""
        
        @self.mcp.tool()
        async def research_juniper_commands(use_case: str, technology_focus: str = "", max_commands: int = 8):
            """
            Research and discover relevant Juniper show commands for a specific use case by searching online documentation.
            
            Args:
                use_case: The specific use case or problem (e.g., "EVPN troubleshooting", "VXLAN verification")
                technology_focus: Specific technology to focus on (e.g., "EVPN", "VXLAN", "BGP", "OSPF")
                max_commands: Maximum number of commands to research and return
            
            Returns:
                Researched commands with descriptions and use case relevance
            """
            # This function would use web search to find commands
            # For now, I'll show the structure - you'd need to implement web search
            
            search_query = f"Juniper {technology_focus} show commands {use_case} troubleshooting site:juniper.net"
            
            # Simulated research results - in practice this would use web_search
            researched_commands = {
                'research_query': search_query,
                'use_case': use_case,
                'technology_focus': technology_focus,
                'discovered_commands': [
                    {
                        'command': 'show evpn database extensive',
                        'description': 'Detailed EVPN database with MAC/IP mappings',
                        'relevance': 'high',
                        'source': 'Juniper documentation',
                        'use_cases': ['EVPN troubleshooting', 'MAC learning issues']
                    },
                    {
                        'command': 'show route table bgp.evpn.0 detail',
                        'description': 'Detailed EVPN routes with path attributes',
                        'relevance': 'high',
                        'source': 'Juniper documentation',
                        'use_cases': ['route advertisement issues', 'EVPN convergence']
                    }
                ],
                '_formatting_hint': '''
        RESEARCH RESULTS - ADHD Format:

        1. **🔍 RESEARCH SUMMARY**:
        ```
        Use Case: {use_case}
        Technology: {technology_focus}
        Commands Found: X relevant commands
        ```

        2. **📚 DISCOVERED COMMANDS**:
        | Command | Purpose | Relevance | Try It? |
        |---------|---------|-----------|---------|
        | show evpn database | EVPN MAC table | 🔥 HIGH | ✅ Yes |
        | show vxlan tunnel | VXLAN tunnels | 🔥 HIGH | ✅ Yes |

        3. **💡 NEXT STEPS**:
        - Use `execute_researched_commands()` to run these
        - Or pick specific commands with `execute_device_command()`

        Format as actionable research results with clear next steps.
                '''
            }
            
            return researched_commands

        @self.mcp.tool()
        async def execute_researched_commands(blueprint_id: str, use_case: str, technology_focus: str = "", device_roles: list = None, max_devices: int = 3):
            """
            Research Juniper commands online, then execute the most relevant ones for the use case.
            
            Args:
                blueprint_id: The blueprint ID to target
                use_case: The specific use case (e.g., "EVPN troubleshooting") 
                technology_focus: Specific technology (e.g., "EVPN", "VXLAN")
                device_roles: Device roles to target (default: spine for control plane, leaf for access)
                max_devices: Maximum devices to execute commands on
            
            Returns:
                Research results and command execution output
            """
            await self.ensure_authenticated()
            
            try:
                # Step 1: Research commands
                research_result = await research_juniper_commands(use_case, technology_focus)
                discovered_commands = research_result.get('discovered_commands', [])
                
                if not discovered_commands:
                    return {
                        'error': f'No relevant commands found for use case: {use_case}',
                        'suggestion': 'Try a more specific use case like "EVPN MAC learning" or "VXLAN tunnel status"'
                    }
                
                # Step 2: Get target devices
                nodes_result = await self._get_blueprint_nodes(blueprint_id)
                if 'error' in str(nodes_result):
                    return nodes_result
                
                # Default device roles based on technology
                if not device_roles:
                    if technology_focus.upper() in ['EVPN', 'BGP']:
                        device_roles = ['spine']  # Control plane
                    elif technology_focus.upper() in ['VXLAN', 'SWITCHING']:
                        device_roles = ['leaf']   # Access layer
                    else:
                        device_roles = ['spine', 'leaf']  # Both
                
                target_devices = self._filter_target_devices(nodes_result, device_roles, None, max_devices)
                
                if not target_devices:
                    return {
                        'error': f'No devices found with roles: {device_roles}',
                        'available_roles': list(set(node.get('role', '') for node in nodes_result.get('nodes', {}).values()))
                    }
                
                # Step 3: Execute discovered commands
                execution_results = {}
                
                # Prioritize high-relevance commands
                high_priority_commands = [cmd for cmd in discovered_commands if cmd.get('relevance') == 'high']
                commands_to_execute = high_priority_commands[:4]  # Limit to top 4 commands
                
                for cmd_info in commands_to_execute:
                    command = cmd_info['command']
                    
                    device_results = {}
                    for device in target_devices:
                        device_id = device.get('system_id')
                        hostname = device.get('hostname', device_id)
                        
                        if device_id:
                            result = await execute_device_command(device_id, command, timeout_seconds=45)
                            device_results[hostname] = {
                                'result': result,
                                'command_purpose': cmd_info['description']
                            }
                    
                    execution_results[command] = {
                        'device_results': device_results,
                        'command_info': cmd_info,
                        'devices_targeted': len(target_devices)
                    }
                
                return {
                    'researched_command_execution': {
                        'research_phase': research_result,
                        'execution_phase': execution_results,
                        'summary': {
                            'use_case': use_case,
                            'technology_focus': technology_focus,
                            'commands_researched': len(discovered_commands),
                            'commands_executed': len(execution_results),
                            'devices_queried': len(target_devices)
                        }
                    },
                    '_formatting_hint': f'''
        DYNAMIC COMMAND RESEARCH & EXECUTION - ADHD Format:

        1. **🎯 RESEARCH MISSION**:
        ```
        Use Case: {use_case}
        Technology: {technology_focus}
        Target: {len(target_devices)} {'/'.join(device_roles)} devices
        ```

        2. **🔬 RESEARCH FINDINGS**:
        - 📚 Commands discovered: {len(discovered_commands)}
        - ⚡ High priority: {len([c for c in discovered_commands if c.get('relevance') == 'high'])}
        - 🎯 Executed: {len(execution_results)}

        3. **📊 EXECUTION RESULTS** (per command):
        | Command | Purpose | Devices | Status |
        |---------|---------|---------|--------|
        | [cmd] | [purpose] | X/Y | ✅/❌ |

        4. **💡 KEY DISCOVERIES**:
        - Most important finding from commands
        - Technology-specific insights
        - Recommendations for next steps

        Present research process, then technical findings with clear visual hierarchy.
                    '''
                }
                
            except Exception as e:
                logger.error(f"Error in researched command execution: {e}")
                return {"error": str(e)}

        @self.mcp.tool()
        async def explore_device_configuration(system_id: str, config_section: str = "", search_terms: list = None):
            """
            Explore device configuration using show configuration commands with optional filtering.
            
            Args:
                system_id: The device system ID to query
                config_section: Specific configuration section (e.g., "protocols bgp", "interfaces", "routing-options")
                search_terms: List of terms to search for in configuration (e.g., ["evpn", "vxlan"])
            
            Returns:
                Device configuration data with intelligent filtering and analysis
            """
            await self.ensure_authenticated()
            
            try:
                # Build the configuration command
                if config_section:
                    command = f"show configuration {config_section}"
                else:
                    command = "show configuration"
                
                # Add search filtering if specified
                if search_terms:
                    # For multiple terms, we'll run separate commands
                    search_commands = []
                    for term in search_terms:
                        search_commands.append(f"show configuration | display set | match {term}")
                else:
                    search_commands = [command]
                
                config_results = {}
                
                for cmd in search_commands:
                    result = await execute_device_command(system_id, cmd, timeout_seconds=30)
                    
                    if isinstance(result, dict) and 'output' in result:
                        # Parse and structure configuration output
                        config_data = self._parse_configuration_output(result['output'], cmd)
                        config_results[cmd] = config_data
                    else:
                        config_results[cmd] = result
                
                return {
                    'device_configuration_exploration': {
                        'device_id': system_id,
                        'config_section': config_section or "full",
                        'search_terms': search_terms or [],
                        'configuration_data': config_results
                    },
                    '_formatting_hint': f'''
        DEVICE CONFIGURATION EXPLORATION - ADHD Format:

        1. **📱 DEVICE INFO**:
        ```
        Device: {system_id}
        Section: {config_section or "Full Configuration"}
        Search: {', '.join(search_terms) if search_terms else "No filtering"}
        ```

        2. **⚙️ CONFIGURATION STRUCTURE**:
        ```
        protocols/
        ├── bgp/
        │   ├── group evpn-spine
        │   └── family evpn
        ├── evpn/
        └── ospf/
        ```

        3. **🔍 KEY CONFIGURATION BLOCKS**:
        <details>
        <summary>📋 BGP Configuration</summary>
        
        ```
        [BGP config here]
        ```
        </details>

        4. **💡 CONFIGURATION INSIGHTS**:
        - EVPN is configured with X settings
        - BGP has Y peer groups
        - Potential issues: [if any]

        Present config in organized, collapsible sections with clear hierarchy.
                    '''
                }
                
            except Exception as e:
                logger.error(f"Error exploring device configuration: {e}")
                return {"error": str(e)}

        def _parse_configuration_output(self, output: str, command: str) -> dict:
            """Parse configuration output into structured format"""
            lines = output.split('\n')
            
            # Basic parsing for different command types
            if 'display set' in command:
                # Set commands - group by hierarchy
                set_commands = [line.strip() for line in lines if line.strip().startswith('set')]
                
                # Group by top-level hierarchy
                config_groups = {}
                for cmd in set_commands:
                    parts = cmd.split()
                    if len(parts) >= 2:
                        top_level = parts[1]  # First element after 'set'
                        if top_level not in config_groups:
                            config_groups[top_level] = []
                        config_groups[top_level].append(cmd)
                
                return {
                    'format': 'set_commands',
                    'total_lines': len(set_commands),
                    'config_sections': config_groups
                }
            
            else:
                # Hierarchical config - basic structure detection
                config_sections = []
                current_section = ""
                
                for line in lines:
                    if line and not line.startswith(' '):
                        current_section = line.strip()
                        config_sections.append(current_section)
                
                return {
                    'format': 'hierarchical',
                    'total_lines': len(lines),
                    'main_sections': config_sections[:20],  # Limit sections
                    'preview': '\n'.join(lines[:50]) if len(lines) > 50 else output
                }

        @self.mcp.tool()
        async def intelligent_show_command_discovery(issue_description: str, device_context: str = ""):
            """
            Use AI reasoning to discover the most relevant Juniper show commands for a specific issue.
            
            Args:
                issue_description: Description of the issue or what needs to be investigated
                device_context: Additional context about the device/network (e.g., "spine switch", "EVPN fabric")
            
            Returns:
                Intelligently selected show commands with reasoning
            """
            
            # Intelligent command mapping based on issue keywords
            command_intelligence = {
                'bgp': {
                    'keywords': ['bgp', 'peering', 'neighbor', 'routing'],
                    'commands': [
                        'show bgp summary',
                        'show bgp neighbor',
                        'show route protocol bgp',
                        'show configuration protocols bgp'
                    ]
                },
                'evpn': {
                    'keywords': ['evpn', 'l2vpn', 'mac learning', 'evi'],
                    'commands': [
                        'show evpn database',
                        'show evpn instance',
                        'show route table bgp.evpn.0',
                        'show configuration protocols evpn'
                    ]
                },
                'vxlan': {
                    'keywords': ['vxlan', 'vtep', 'tunnel', 'overlay'],
                    'commands': [
                        'show interfaces vtep',
                        'show vxlan tunnel',
                        'show vxlan statistics',
                        'show configuration vlans'
                    ]
                },
                'interface': {
                    'keywords': ['interface', 'port', 'link', 'down', 'flapping'],
                    'commands': [
                        'show interfaces terse',
                        'show interfaces extensive',
                        'show interfaces diagnostics optics',
                        'show configuration interfaces'
                    ]
                },
                'routing': {
                    'keywords': ['route', 'reachability', 'convergence', 'forwarding'],
                    'commands': [
                        'show route summary',
                        'show route extensive',
                        'show isis adjacency',
                        'show ospf neighbor'
                    ]
                }
            }
            
            # Analyze issue description to find relevant commands
            issue_lower = issue_description.lower()
            relevant_commands = []
            
            for category, info in command_intelligence.items():
                relevance_score = 0
                for keyword in info['keywords']:
                    if keyword in issue_lower:
                        relevance_score += 1
                
                if relevance_score > 0:
                    for cmd in info['commands']:
                        relevant_commands.append({
                            'command': cmd,
                            'category': category,
                            'relevance_score': relevance_score,
                            'reasoning': f"Matches {relevance_score} keywords related to {category}"
                        })
            
            # Sort by relevance and return top commands
            relevant_commands.sort(key=lambda x: x['relevance_score'], reverse=True)
            
            return {
                'intelligent_command_discovery': {
                    'issue_analyzed': issue_description,
                    'device_context': device_context,
                    'recommended_commands': relevant_commands[:8],  # Top 8 commands
                    'categories_matched': list(set(cmd['category'] for cmd in relevant_commands))
                },
                '_formatting_hint': '''
        INTELLIGENT COMMAND DISCOVERY - ADHD Format:

        1. **🧠 ANALYSIS SUMMARY**:
        ```
        Issue: {issue_description}
        Context: {device_context}
        Categories: X matched
        Commands: Y recommended
        ```

        2. **🎯 RECOMMENDED COMMANDS**:
        | Priority | Command | Purpose | Reasoning |
        |----------|---------|---------|-----------|
        | 🔥 HIGH | show bgp summary | BGP status | Matches issue keywords |

        3. **🚀 QUICK ACTIONS**:
        - Use `execute_device_command()` for single device
        - Use `execute_fabric_command()` for multiple devices
        - Use `explore_device_configuration()` for config analysis

        4. **💡 DISCOVERY LOGIC**:
        - Analyzed issue for key technologies
        - Matched against Juniper command database
        - Ranked by relevance to your specific problem

        Present as actionable intelligence with clear next steps.
                '''
            }


        @self.mcp.tool()
        async def execute_device_command(system_id: str, command_text: str, output_format: str = "text", timeout_seconds: int = 30):
            """
            Execute a show command on a specific network device and wait for results.
            
            Args:
                system_id: The system ID of the target device (from blueprint nodes)
                command_text: The Juniper show command to execute (e.g. 'show version')
                output_format: Format for output - text, json, or xml
                timeout_seconds: Maximum time to wait for command completion
            
            Returns:
                Command output from the device or error information
            """
            await self.ensure_authenticated()
            
            try:
                # Step 1: Submit the command
                submit_result = await self._submit_device_command(system_id, command_text, output_format)
                if 'error' in submit_result:
                    return submit_result
                
                request_id = submit_result.get('request_id')
                if not request_id:
                    return {"error": "No request_id returned from command submission"}
                
                # Step 2: Poll for results with timeout
                result = await self._poll_command_result(request_id, timeout_seconds)
                
                # Add formatting hints based on command type
                if isinstance(result, dict) and 'output' in result:
                    result['_formatting_hint'] = f'''
DEVICE COMMAND OUTPUT for ADHD users:

1. **📱 COMMAND SUMMARY**:
   ```
   Device: {system_id}
   Command: {command_text}
   Format: {output_format}
   Status: ✅ Success / ❌ Failed
   ```

2. **🎯 KEY FINDINGS** (extract top 3-5 important items):
   - Most important finding
   - Second key point  
   - Third notable item

3. **📊 RAW OUTPUT** (in collapsible code block):
   ```
   [First 20 lines of actual output]
   ... [indicate if truncated]
   ```

4. **💡 INTERPRETATION**:
   - What this means for network health
   - Any action items needed

Keep technical details in code blocks, highlight key numbers with backticks.
                    '''
                    result['_command_info'] = {
                        'device': system_id,
                        'command': command_text,
                        'format': output_format
                    }
                
                return result
                
            except Exception as e:
                logger.error(f"Error executing device command: {e}")
                return {"error": str(e)}
        
        @self.mcp.tool()
        async def execute_fabric_command(blueprint_id: str, command_text: str, device_roles: list = None, device_types: list = None, max_devices: int = 10):
            """
            Execute a show command across multiple devices in the fabric.
            
            Args:
                blueprint_id: The blueprint ID to target
                command_text: The Juniper show command to execute
                device_roles: List of device roles to target (e.g. ['spine', 'leaf'])
                device_types: List of device types to target (e.g. ['switch'])
                max_devices: Maximum number of devices to query (to prevent overload)
            
            Returns:
                Dictionary of results keyed by device hostname/ID
            """
            await self.ensure_authenticated()
            
            try:
                # Get blueprint nodes to find target devices
                nodes_result = await self._get_blueprint_nodes(blueprint_id)
                if 'error' in nodes_result:
                    return nodes_result
                
                # Filter devices based on criteria
                target_devices = self._filter_target_devices(
                    nodes_result, device_roles, device_types, max_devices
                )
                
                if not target_devices:
                    return {
                        "error": "No devices found matching the specified criteria",
                        "criteria": {
                            "roles": device_roles,
                            "types": device_types
                        }
                    }
                
                # Execute command on all target devices
                results = {}
                for device in target_devices:
                    device_id = device.get('system_id') or device.get('id')
                    hostname = device.get('hostname') or device.get('label', device_id)
                    
                    result = await execute_device_command(device_id, command_text)
                    results[hostname] = result
                
                return {
                    'fabric_command_results': results,
                    'command': command_text,
                    'devices_queried': len(results),
                    '_formatting_hint': f'''
FABRIC-WIDE COMMAND RESULTS - ADHD Format:

1. **📊 EXECUTION SUMMARY**:
   ```
   Command: {command_text}
   Devices: {len(results)} queried
   Success: X/Y completed
   ```

2. **🎯 QUICK COMPARISON TABLE**:
   | Device | Status | Key Metric | Notes |
   |--------|--------|------------|-------|
   | Spine-1 | ✅ | Value | Normal |
   | Spine-2 | ⚠️  | Value | Check |

3. **🔍 DETAILED RESULTS** (collapsible per device):
   <details>
   <summary>📱 Device-1 Details</summary>
   
   ```
   [Command output here]
   ```
   </details>

4. **🚨 INCONSISTENCIES** (if any):
   - Device X shows different value
   - Device Y has error condition

Focus on differences between devices and highlight anomalies.
                    '''
                }
                
            except Exception as e:
                logger.error(f"Error executing fabric command: {e}")
                return {"error": str(e)}
        
        @self.mcp.tool()
        async def analyze_fabric_health_comprehensive(blueprint_id: str):
            """
            Comprehensive fabric health analysis using blueprint data first, device commands only when necessary.
            
            Args:
                blueprint_id: The blueprint ID to analyze
            
            Returns:
                Multi-layered health analysis starting with blueprint intent, then device reality
            """
            await self.ensure_authenticated()
            
            health_analysis = {}
            
            # Phase 1: Blueprint Intent Analysis (fast, no device load)
            try:
                # Get high-level blueprint status
                blueprint_result = await self.functions['get_api_blueprints_blueprint_id'].execute(
                    self.auth_config, blueprint_id=blueprint_id
                )
                # Summarize blueprint data instead of returning raw data
                health_analysis['blueprint_intent'] = self._summarize_blueprint_data(blueprint_result)
                
                # Get blueprint errors (intent vs reality mismatches)
                errors_result = await self.functions['get_api_blueprints_blueprint_id_errors'].execute(
                    self.auth_config, blueprint_id=blueprint_id
                )
                health_analysis['intent_violations'] = self._summarize_errors_data(errors_result)
                
                # Get node status from blueprint perspective - SUMMARIZED
                nodes_result = await self.functions['get_api_blueprints_blueprint_id_nodes'].execute(
                    self.auth_config, blueprint_id=blueprint_id
                )
                health_analysis['node_status'] = self._summarize_nodes_data(nodes_result)
                
                logger.info("Phase 1 complete: Blueprint intent analysis")
                
                # Phase 2: Selective Device Commands (only if blueprint shows issues)
                device_commands_needed = self._assess_device_command_necessity(
                    blueprint_result, errors_result, nodes_result
                )
                
                if device_commands_needed:
                    logger.info(f"Phase 2: Running {len(device_commands_needed)} targeted device commands")
                    device_analysis = {}
                    
                    for command_info in device_commands_needed:
                        command = command_info['command']
                        target_devices = command_info['target_devices']
                        reason = command_info['reason']
                        
                        logger.info(f"Running '{command}' on {len(target_devices)} devices: {reason}")
                        
                        # Run command only on specific devices that need it
                        device_results = {}
                        for device in target_devices[:2]:  # Limit to 2 devices max per command
                            device_id = device.get('system_id')
                            if device_id:
                                result = await execute_device_command(device_id, command, timeout_seconds=45)
                                # Summarize device command output too
                                device_results[device.get('hostname', device_id)] = self._summarize_device_output(result, command)
                        
                        device_analysis[command] = {
                            'results': device_results,
                            'reason': reason,
                            'devices_targeted': len(target_devices)
                        }
                    
                    health_analysis['device_verification'] = device_analysis
                else:
                    logger.info("Phase 2: No device commands needed - blueprint data sufficient")
                    health_analysis['device_verification'] = {
                        'status': 'not_needed',
                        'reason': 'Blueprint data shows no issues requiring device-level verification'
                    }
                
            except Exception as e:
                logger.error(f"Error in comprehensive health analysis: {e}")
                return {"error": str(e)}
            
            return {
                'comprehensive_health_analysis': health_analysis,
                '_formatting_hint': '''
CRITICAL: Format this data for ADHD users - use visual structure and clear sections:

1. **EXECUTIVE SUMMARY** (top of response):
   - 🟢/🟡/🔴 Overall health status
   - Key numbers in badges: `Total Devices: 12` `Errors: 3` `Alarms: 1`

2. **BLUEPRINT INTENT** section:
   ```
   📋 BLUEPRINT STATUS
   ├── ID: {blueprint_id}
   ├── Status: {status}
   ├── Design: {design_type}
   └── Anomalies: {count}
   ```

3. **ISSUES FOUND** (if any):
   | Severity | Count | Type | Example |
   |----------|-------|------|---------|
   | 🔴 Critical | 2 | BGP Down | Peer 1.2.3.4 |
   | 🟡 Warning | 5 | Interface | Port down |

4. **DEVICE STATUS** as ASCII chart:
   ```
   DEVICE TYPES
   ████████████ Spine (4)
   ██████ Leaf (8)  
   ██ Border (2)
   ```

5. **RECOMMENDATIONS** as numbered list:
   1. ⚡ **URGENT**: Fix BGP peer X.X.X.X
   2. 🔧 **ACTION**: Check interface Y
   3. 📊 **MONITOR**: Watch anomaly trend

Use emojis, code blocks, tables, and visual separators. Keep each section under 10 lines.
                ''',
                '_analysis_type': 'comprehensive_health',
                '_visual_format': 'adhd_friendly'
            }
    
    def _summarize_blueprint_data(self, blueprint_result):
        """Summarize blueprint data to key metrics only"""
        if not isinstance(blueprint_result, dict):
            return {"summary": "Blueprint data unavailable"}
        
        summary = {
            'blueprint_id': blueprint_result.get('id', 'Unknown'),
            'status': blueprint_result.get('status', 'Unknown'),
            'design_type': blueprint_result.get('design', {}).get('type', 'Unknown'),
            'last_modified': blueprint_result.get('last_modified_at', 'Unknown')
        }
        
        # Add anomaly counts if available
        anomaly_counts = blueprint_result.get('anomaly_counts', {})
        if anomaly_counts:
            total_anomalies = sum(anomaly_counts.values()) if isinstance(anomaly_counts, dict) else 0
            summary['total_anomalies'] = total_anomalies
            summary['anomaly_types'] = list(anomaly_counts.keys()) if isinstance(anomaly_counts, dict) else []
        
        return summary
    
    def _summarize_errors_data(self, errors_result):
        """Summarize errors data to counts and types only"""
        if not isinstance(errors_result, dict):
            return {"summary": "No error data available"}
        
        error_summary = {
            'total_errors': 0,
            'error_categories': {},
            'critical_errors': []
        }
        
        for error_type, errors in errors_result.items():
            if isinstance(errors, list):
                error_count = len(errors)
                error_summary['total_errors'] += error_count
                error_summary['error_categories'][error_type] = error_count
                
                # Keep only first 3 errors as examples
                if errors and error_count > 0:
                    sample_errors = errors[:3]
                    error_summary['critical_errors'].extend([
                        {'type': error_type, 'example': str(error)[:200] + '...' if len(str(error)) > 200 else str(error)}
                        for error in sample_errors
                    ])
        
        return error_summary
    
    def _summarize_nodes_data(self, nodes_result):
        """Summarize nodes data to counts and status overview"""
        if not isinstance(nodes_result, dict) or 'nodes' not in nodes_result:
            return {"summary": "No nodes data available"}
        
        nodes = nodes_result['nodes']
        node_summary = {
            'total_nodes': len(nodes),
            'node_types': {},
            'node_roles': {},
            'devices_with_issues': [],
            'manageable_devices': 0
        }
        
        for node_id, node in nodes.items():
            # Count by type
            node_type = node.get('type', 'unknown')
            node_summary['node_types'][node_type] = node_summary['node_types'].get(node_type, 0) + 1
            
            # Count by role
            node_role = node.get('role', 'unknown')
            node_summary['node_roles'][node_role] = node_summary['node_roles'].get(node_role, 0) + 1
            
            # Check for manageable devices
            if node.get('system_id'):
                node_summary['manageable_devices'] += 1
            
            # Check for devices with issues (keep only summary)
            status = node.get('status', 'unknown')
            if status.lower() in ['error', 'warning', 'down', 'unreachable']:
                node_summary['devices_with_issues'].append({
                    'id': node_id[:16] + '...' if len(node_id) > 16 else node_id,  # Truncate long IDs
                    'type': node_type,
                    'role': node_role,
                    'status': status,
                    'hostname': node.get('hostname', node.get('label', 'Unknown'))[:30]  # Limit hostname length
                })
        
        # Limit devices_with_issues to first 10
        node_summary['devices_with_issues'] = node_summary['devices_with_issues'][:10]
        
        return node_summary
    
    def _summarize_device_output(self, device_result, command):
        """Summarize device command output to key information only"""
        if not isinstance(device_result, dict):
            return {"summary": f"Device command '{command}' failed or returned unexpected format"}
        
        if 'error' in device_result:
            return {"error": device_result['error']}
        
        output = device_result.get('output', '')
        if not output:
            return {"summary": f"Command '{command}' returned no output"}
        
        # Summarize based on command type
        if 'show bgp summary' in command:
            return self._summarize_bgp_output(output)
        elif 'show system alarms' in command:
            return self._summarize_alarms_output(output)
        elif 'route table' in command:
            return self._summarize_route_output(output)
        else:
            # Generic summarization - just first and last few lines
            lines = output.split('\n')
            if len(lines) > 20:
                summary_output = '\n'.join(lines[:10]) + '\n... [truncated] ...\n' + '\n'.join(lines[-5:])
            else:
                summary_output = output
            
            return {
                'command': command,
                'output_preview': summary_output[:1000] + '...' if len(summary_output) > 1000 else summary_output,
                'line_count': len(lines)
            }
    
    def _summarize_bgp_output(self, output):
        """Extract key BGP summary information"""
        lines = output.split('\n')
        summary = {
            'bgp_summary': 'BGP status extracted',
            'peer_count': 0,
            'established_peers': 0,
            'down_peers': []
        }
        
        for line in lines:
            if 'Established' in line:
                summary['established_peers'] += 1
            elif any(state in line for state in ['Idle', 'Connect', 'Active']):
                # Extract peer IP/info for down peers
                parts = line.split()
                if parts:
                    summary['down_peers'].append(parts[0])  # Usually the first part is peer IP
            summary['peer_count'] += 1
        
        return summary
    
    def _summarize_alarms_output(self, output):
        """Extract key alarm information"""
        lines = output.split('\n')
        alarms = []
        
        for line in lines:
            if line.strip() and not line.startswith('No alarms') and 'alarm' in line.lower():
                alarms.append(line.strip()[:100])  # Limit alarm description length
        
        return {
            'alarm_summary': f"{len(alarms)} alarms detected" if alarms else "No active alarms",
            'active_alarms': alarms[:5]  # Limit to first 5 alarms
        }
    
    def _summarize_route_output(self, output):
        """Extract key routing information"""
        lines = output.split('\n')
        route_count = 0
        
        for line in lines:
            if '/' in line and ('via' in line or 'to' in line):  # Likely a route line
                route_count += 1
        
        return {
            'route_summary': f"Approximately {route_count} routes found",
            'output_preview': '\n'.join(lines[:15]) if len(lines) > 15 else output
        }
        
        @self.mcp.tool()
        async def intelligent_evpn_analysis(blueprint_id: str):
            """
            Intelligent EVPN analysis: blueprint intent first, device commands only for operational data.
            
            Args:
                blueprint_id: The blueprint ID to analyze
            
            Returns:
                EVPN analysis prioritizing blueprint data, with selective device verification
            """
            await self.ensure_authenticated()
            
            evpn_analysis = {}
            
            # Phase 1: Blueprint EVPN Intent
            try:
                # Get blueprint design and EVPN configuration
                blueprint_result = await self.functions['get_api_blueprints_blueprint_id'].execute(
                    self.auth_config, blueprint_id=blueprint_id
                )
                
                # Get any EVPN-related errors from blueprint
                errors_result = await self.functions['get_api_blueprints_blueprint_id_errors'].execute(
                    self.auth_config, blueprint_id=blueprint_id
                )
                
                evpn_analysis['evpn_intent'] = {
                    'blueprint_design': blueprint_result,
                    'intent_violations': errors_result
                }
                
                # Phase 2: Selective Device Commands (only for operational state)
                # These are things blueprint can't tell us
                spine_devices = self._get_spine_devices_from_nodes(blueprint_id)
                
                if spine_devices:
                    logger.info("Getting operational EVPN state from spine switches")
                    
                    operational_commands = [
                        {
                            'command': 'show route table bgp.evpn.0 summary',
                            'reason': 'Current EVPN route counts (not in blueprint)',
                            'priority': 'high'
                        },
                        {
                            'command': 'show bgp summary',
                            'reason': 'BGP session state (real-time status)',
                            'priority': 'high'
                        }
                    ]
                    
                    # Only run on 2 spine switches max
                    limited_spines = spine_devices[:2]
                    operational_data = {}
                    
                    for cmd_info in operational_commands:
                        command = cmd_info['command']
                        device_results = {}
                        
                        for device in limited_spines:
                            device_id = device.get('system_id')
                            if device_id:
                                result = await execute_device_command(device_id, command, timeout_seconds=30)
                                device_results[device.get('hostname', device_id)] = result
                        
                        operational_data[command] = {
                            'results': device_results,
                            'reason': cmd_info['reason']
                        }
                    
                    evpn_analysis['operational_state'] = operational_data
                
            except Exception as e:
                logger.error(f"Error in intelligent EVPN analysis: {e}")
                return {"error": str(e)}
            
            return {
                'intelligent_evpn_analysis': evpn_analysis,
                '_formatting_hint': '''
ADHD-FRIENDLY EVPN ANALYSIS FORMAT:

1. **🎯 QUICK STATUS**:
   ```
   EVPN HEALTH: 🟢 GOOD / 🟡 ISSUES / 🔴 CRITICAL
   BGP Peers: X/Y UP    Routes: #### Active
   ```

2. **📊 EVPN METRICS TABLE**:
   | Metric | Expected | Actual | Status |
   |--------|----------|--------|--------|
   | BGP Sessions | 6 | 4 | 🟡 |
   | EVPN Routes | ~200 | 187 | 🟢 |

3. **🏗️ BLUEPRINT vs REALITY**:
   ```
   INTENT                    REALITY
   ├── 4 Spine switches  →  ✅ 4 Active
   ├── EVPN Enabled      →  ✅ Running  
   └── BGP Peering       →  ⚠️  2 Down
   ```

4. **🚨 ACTION ITEMS** (if any):
   - [ ] **HIGH**: Restore BGP peer 10.1.1.1
   - [ ] **MED**: Check route advertisement
   - [ ] **LOW**: Monitor convergence time

Keep sections bite-sized with clear visual hierarchy.
                ''',
                '_analysis_type': 'intelligent_evpn',
                '_visual_format': 'adhd_friendly'
            }
        
        @self.mcp.tool()
        async def blueprint_first_troubleshooting(blueprint_id: str, issue_description: str):
            """
            Troubleshooting that starts with blueprint analysis and only uses device commands when blueprint data is insufficient.
            
            Args:
                blueprint_id: The blueprint ID to troubleshoot
                issue_description: Description of the issue to investigate
            
            Returns:
                Troubleshooting analysis that escalates from blueprint to device level only when needed
            """
            await self.ensure_authenticated()
            
            troubleshooting_analysis = {}
            
            try:
                # Phase 1: Blueprint Analysis (always first)
                blueprint_data = await self._comprehensive_blueprint_analysis(blueprint_id)
                troubleshooting_analysis['blueprint_analysis'] = blueprint_data
                
                # Phase 2: Determine if device commands are needed
                device_investigation = self._determine_device_investigation_needs(
                    blueprint_data, issue_description
                )
                
                if device_investigation['needed']:
                    logger.info(f"Device investigation needed: {device_investigation['reason']}")
                    
                    targeted_commands = device_investigation['commands']
                    target_devices = device_investigation['devices']
                    
                    device_results = {}
                    for cmd_info in targeted_commands:
                        command = cmd_info['command']
                        max_devices = cmd_info.get('max_devices', 2)
                        
                        # Limit device load
                        limited_devices = target_devices[:max_devices]
                        
                        command_results = {}
                        for device in limited_devices:
                            device_id = device.get('system_id')
                            if device_id:
                                result = await execute_device_command(device_id, command, timeout_seconds=40)
                                command_results[device.get('hostname', device_id)] = result
                        
                        device_results[command] = {
                            'results': command_results,
                            'reason': cmd_info['reason']
                        }
                    
                    troubleshooting_analysis['device_investigation'] = device_results
                else:
                    troubleshooting_analysis['device_investigation'] = {
                        'status': 'not_needed',
                        'reason': device_investigation['reason']
                    }
                
            except Exception as e:
                logger.error(f"Error in blueprint-first troubleshooting: {e}")
                return {"error": str(e)}
            
            return {
                'blueprint_first_troubleshooting': troubleshooting_analysis,
                'issue_description': issue_description,
                '_formatting_hint': f'''
ADHD-OPTIMIZED TROUBLESHOOTING REPORT for: "{issue_description}"

1. **⚡ IMMEDIATE FINDINGS**:
   ```
   🔍 ISSUE: {issue_description}
   📋 BLUEPRINT: [Quick status]
   🖥️  DEVICES: [If checked]
   ```

2. **🎯 ROOT CAUSE ANALYSIS**:
   | Layer | Status | Evidence | Action |
   |-------|--------|----------|---------|
   | Intent | ✅/❌ | Blueprint shows... | None/Fix config |
   | Physical | ✅/❌ | Device shows... | Check hardware |
   | Protocol | ✅/❌ | BGP/OSPF status | Restart/Config |

3. **📈 EVIDENCE TREE**:
   ```
   PROBLEM: {issue_description}
   ├── 📋 Blueprint Check
   │   ├── ✅ Configuration OK
   │   └── ❌ Policy mismatch → FIX CONFIG
   └── 🖥️  Device Check  
       ├── ⚠️  BGP session down
       └── 🔴 Interface flapping → CHECK CABLE
   ```

4. **✅ STEP-BY-STEP FIX**:
   1. **FIRST**: Check X (takes 2 min)
   2. **THEN**: Verify Y (takes 5 min)  
   3. **FINALLY**: Test Z (takes 1 min)

Use clear visual hierarchy with time estimates for each action.
                ''',
                '_analysis_type': 'blueprint_first_troubleshooting',
                '_visual_format': 'adhd_friendly'
            }
        
        @self.mcp.tool()
        async def check_bgp_status(blueprint_id: str, device_roles: list = ['spine']):
            """
            Check BGP peering status across specified devices in the fabric.
            
            Args:
                blueprint_id: The blueprint ID to check
                device_roles: List of device roles to check (default: spine switches)
            
            Returns:
                BGP status summary across all specified devices
            """
            await self.ensure_authenticated()
            
            bgp_commands = [
                'show bgp summary',
                'show bgp neighbor'
            ]
            
            bgp_results = {}
            for command in bgp_commands:
                result = await execute_fabric_command(
                    blueprint_id=blueprint_id,
                    command_text=command,
                    device_roles=device_roles,
                    max_devices=10
                )
                bgp_results[command] = result
            
            return {
                'bgp_status_analysis': bgp_results,
                '_formatting_hint': 'Summarize BGP peer states, session counts, and highlight any down peers or issues.',
                '_analysis_type': 'bgp_status'
            }
        
        @self.mcp.tool()
        async def check_interface_status(blueprint_id: str, interface_type: str = 'all', device_roles: list = None):
            """
            Check interface status and statistics across the fabric.
            
            Args:
                blueprint_id: The blueprint ID to check
                interface_type: Type of interfaces to focus on ('all', 'fabric', 'access', 'management')
                device_roles: List of device roles to check (default: all network devices)
            
            Returns:
                Interface status summary with any down or problematic interfaces highlighted
            """
            await self.ensure_authenticated()
            
            # Choose commands based on interface type
            if interface_type == 'fabric':
                commands = ['show interfaces terse | match "et-|xe-"', 'show lacp interfaces']
            elif interface_type == 'access':
                commands = ['show interfaces terse | match "ge-|fe-"', 'show ethernet-switching table']
            elif interface_type == 'management':
                commands = ['show interfaces terse | match "me0|em0|fxp0"']
            else:  # all
                commands = ['show interfaces terse', 'show interfaces extensive | match "Physical|Description|Link|Input|Output"']
            
            interface_results = {}
            for command in commands:
                result = await execute_fabric_command(
                    blueprint_id=blueprint_id,
                    command_text=command,
                    device_roles=device_roles or ['spine', 'leaf'],
                    max_devices=15
                )
                interface_results[command] = result
            
            return {
                'interface_status_analysis': interface_results,
                'interface_type_focus': interface_type,
                '_formatting_hint': 'Highlight any down interfaces, error counters, or performance issues. Group by device role.',
                '_analysis_type': 'interface_status'
            }
        
        @self.mcp.tool()
        async def check_routing_table(blueprint_id: str, route_type: str = 'summary', device_roles: list = ['spine']):
            """
            Analyze routing tables and protocol status across the fabric.
            
            Args:
                blueprint_id: The blueprint ID to check
                route_type: Type of routing analysis ('summary', 'bgp', 'isis', 'evpn')
                device_roles: List of device roles to check
            
            Returns:
                Routing analysis with route counts and protocol health
            """
            await self.ensure_authenticated()
            
            # Choose commands based on route type
            if route_type == 'bgp':
                commands = ['show route protocol bgp', 'show bgp summary']
            elif route_type == 'isis':
                commands = ['show isis adjacency', 'show isis database']
            elif route_type == 'evpn':
                commands = ['show route table bgp.evpn.0', 'show evpn instance']
            else:  # summary
                commands = ['show route summary', 'show route protocol']
            
            routing_results = {}
            for command in commands:
                result = await execute_fabric_command(
                    blueprint_id=blueprint_id,
                    command_text=command,
                    device_roles=device_roles,
                    max_devices=8
                )
                routing_results[command] = result
            
            return {
                'routing_analysis': routing_results,
                'route_type_focus': route_type,
                '_formatting_hint': 'Summarize route counts, protocol adjacencies, and highlight any missing or unexpected routes.',
                '_analysis_type': 'routing_status'
            }
        
        @self.mcp.tool()
        async def diagnose_connectivity_issue(blueprint_id: str, source_device: str = None, destination_device: str = None):
            """
            Diagnose connectivity issues between devices or across the fabric.
            
            Args:
                blueprint_id: The blueprint ID to diagnose
                source_device: Source device system_id (optional - will check fabric-wide if not specified)
                destination_device: Destination device system_id (optional)
            
            Returns:
                Comprehensive connectivity diagnosis with recommended actions
            """
            await self.ensure_authenticated()
            
            # Commands for connectivity diagnosis
            diagnostic_commands = [
                'show interfaces terse',
                'show bgp summary',
                'show isis adjacency',
                'show route summary'
            ]
            
            if source_device:
                # Focus on specific device
                connectivity_results = {}
                for command in diagnostic_commands:
                    result = await execute_device_command(source_device, command)
                    connectivity_results[command] = result
                
                # Add ping/traceroute if destination specified
                if destination_device:
                    ping_result = await execute_device_command(
                        source_device, 
                        f"ping {destination_device} count 3"
                    )
                    connectivity_results['ping_test'] = ping_result
            else:
                # Fabric-wide diagnosis
                connectivity_results = {}
                for command in diagnostic_commands:
                    result = await execute_fabric_command(
                        blueprint_id=blueprint_id,
                        command_text=command,
                        device_roles=['spine', 'leaf'],
                        max_devices=10
                    )
                    connectivity_results[command] = result
            
            return {
                'connectivity_diagnosis': connectivity_results,
                'source_device': source_device,
                'destination_device': destination_device,
                '_formatting_hint': 'Identify connectivity issues: down interfaces, missing adjacencies, routing problems. Provide step-by-step troubleshooting recommendations.',
                '_analysis_type': 'connectivity_diagnosis'
            }
        
        @self.mcp.tool()
        async def get_device_information(blueprint_id: str, system_id: str):
            """
            Get comprehensive information about a specific network device.
            
            Args:
                blueprint_id: The blueprint ID
                system_id: The system ID of the device to examine
            
            Returns:
                Detailed device information including hardware, software, and operational status
            """
            await self.ensure_authenticated()
            
            device_info_commands = [
                'show version',
                'show chassis hardware',
                'show system uptime',
                'show system alarms',
                'show interfaces terse'
            ]
            
            device_results = {}
            for command in device_info_commands:
                result = await execute_device_command(system_id, command)
                device_results[command] = result
            
            return {
                'device_information': device_results,
                'system_id': system_id,
                '_formatting_hint': 'Present device details in organized sections: Hardware, Software, Status, Alarms, and Interface Summary.',
                '_analysis_type': 'device_info'
            }
    
    async def _submit_device_command(self, system_id: str, command_text: str, output_format: str = "text"):
        """Submit a command to a device"""
        submit_payload = {
            "system_id": system_id,
            "command_text": command_text,
            "output_format": output_format
        }
        
        # Use the existing function registration to call the API
        return await self.functions['post_api_telemetry_fetchcmd'].execute(
            self.auth_config, **submit_payload
        )
    
    async def _poll_command_result(self, request_id: str, timeout_seconds: int = 30):
        """Poll for command result with timeout and cleanup"""
        import asyncio
        
        poll_interval = 2  # Poll every 2 seconds to be less aggressive
        max_attempts = timeout_seconds // poll_interval
        
        logger.info(f"Polling for command result {request_id}, max attempts: {max_attempts}")
        
        for attempt in range(max_attempts):
            try:
                result = await self.functions['get_api_telemetry_fetchcmd_request_id'].execute(
                    self.auth_config, request_id=request_id, keep=True
                )
                
                # Check if we got a successful result
                if isinstance(result, dict) and result.get('result') == 'success':
                    logger.info(f"Command {request_id} completed successfully")
                    await self._cleanup_command_result(request_id)
                    return result
                elif isinstance(result, dict) and 'output' in result:
                    logger.info(f"Command {request_id} completed with output")
                    await self._cleanup_command_result(request_id)
                    return result
                elif isinstance(result, dict) and result.get('error'):
                    logger.error(f"Command {request_id} failed with error: {result.get('error')}")
                    await self._cleanup_command_result(request_id)
                    return result
                
                # Check for expected "still processing" responses
                if isinstance(result, dict):
                    # 404 - command still running
                    if 'message' in result and 'not found' in result['message'].lower():
                        logger.debug(f"Command {request_id} still running (404), attempt {attempt + 1}/{max_attempts}")
                        await asyncio.sleep(poll_interval)
                        continue
                    
                    # 403 - often means command is still being processed or authentication issue
                    if result.get('status_code') == 403:
                        logger.debug(f"Command {request_id} got 403 (likely still processing), attempt {attempt + 1}/{max_attempts}")
                        await asyncio.sleep(poll_interval)
                        continue
                    
                    # Check for error field with 403 status
                    error_msg = result.get('error', '').lower()
                    if '403' in error_msg or 'forbidden' in error_msg:
                        logger.debug(f"Command {request_id} got 403 error (likely still processing), attempt {attempt + 1}/{max_attempts}")
                        await asyncio.sleep(poll_interval)
                        continue
                
                # Unexpected response - log and continue
                logger.warning(f"Unexpected poll response for {request_id}: {result}")
                await asyncio.sleep(poll_interval)
                
            except Exception as e:
                logger.error(f"Error polling for result {request_id}: {e}")
                if attempt == max_attempts - 1:  # Last attempt
                    await self._cleanup_command_result(request_id)
                    return {"error": f"Timeout waiting for command result after {timeout_seconds}s: {e}"}
                await asyncio.sleep(poll_interval)
        
        # Timeout - try to clean up
        await self._cleanup_command_result(request_id)
        return {"error": f"Command timeout after {timeout_seconds} seconds. The command may still be running on the device."}
    
    async def _cleanup_command_result(self, request_id: str):
        """Clean up command result from server cache"""
        try:
            await self.functions['delete_api_telemetry_fetchcmd_request_id'].execute(
                self.auth_config, request_id=request_id
            )
            logger.debug(f"Cleaned up command result {request_id}")
        except Exception as e:
            logger.debug(f"Cleanup failed for {request_id} (this is normal): {e}")
            # Don't fail the main operation if cleanup fails
    
    async def _get_blueprint_nodes(self, blueprint_id: str):
        """Get nodes from blueprint"""
        return await self.functions['get_api_blueprints_blueprint_id_nodes'].execute(
            self.auth_config, blueprint_id=blueprint_id
        )
    
    def _assess_device_command_necessity(self, blueprint_result, errors_result, nodes_result):
        """Assess whether device commands are needed based on blueprint data"""
        device_commands_needed = []
        
        # Check if there are blueprint errors that might need device verification
        if isinstance(errors_result, dict):
            error_count = 0
            for error_type, errors in errors_result.items():
                if isinstance(errors, list):
                    error_count += len(errors)
            
            if error_count > 0:
                # Get spine devices for BGP/EVPN checks
                spine_devices = self._extract_spine_devices(nodes_result)
                if spine_devices:
                    device_commands_needed.append({
                        'command': 'show bgp summary',
                        'target_devices': spine_devices,
                        'reason': f'Blueprint shows {error_count} errors - verifying BGP sessions'
                    })
        
        # Check for specific node issues
        if isinstance(nodes_result, dict) and 'nodes' in nodes_result:
            problem_devices = []
            for node_id, node in nodes_result['nodes'].items():
                # Look for nodes that might have operational issues
                if node.get('status', '').lower() in ['error', 'warning', 'down']:
                    if node.get('system_id'):  # Only if it's a manageable device
                        problem_devices.append(node)
            
            if problem_devices:
                device_commands_needed.append({
                    'command': 'show system alarms',
                    'target_devices': problem_devices[:2],  # Limit to 2 devices
                    'reason': f'Blueprint shows {len(problem_devices)} devices with status issues'
                })
        
        return device_commands_needed
    
    def _extract_spine_devices(self, nodes_result):
        """Extract spine devices from nodes result"""
        spine_devices = []
        if isinstance(nodes_result, dict) and 'nodes' in nodes_result:
            for node_id, node in nodes_result['nodes'].items():
                if 'spine' in node.get('role', '').lower() and node.get('system_id'):
                    node['id'] = node_id
                    spine_devices.append(node)
        return spine_devices
    
    async def _get_spine_devices_from_nodes(self, blueprint_id: str):
        """Get spine devices for the blueprint"""
        try:
            nodes_result = await self.functions['get_api_blueprints_blueprint_id_nodes'].execute(
                self.auth_config, blueprint_id=blueprint_id
            )
            return self._extract_spine_devices(nodes_result)
        except Exception as e:
            logger.error(f"Error getting spine devices: {e}")
            return []
    
    async def _comprehensive_blueprint_analysis(self, blueprint_id: str):
        """Get comprehensive blueprint data without device commands"""
        blueprint_analysis = {}
        
        try:
            # Core blueprint info
            blueprint_result = await self.functions['get_api_blueprints_blueprint_id'].execute(
                self.auth_config, blueprint_id=blueprint_id
            )
            blueprint_analysis['blueprint_info'] = blueprint_result
            
            # Blueprint errors/violations
            errors_result = await self.functions['get_api_blueprints_blueprint_id_errors'].execute(
                self.auth_config, blueprint_id=blueprint_id
            )
            blueprint_analysis['errors'] = errors_result
            
            # Node status and configuration
            nodes_result = await self.functions['get_api_blueprints_blueprint_id_nodes'].execute(
                self.auth_config, blueprint_id=blueprint_id
            )
            blueprint_analysis['nodes'] = nodes_result
            
        except Exception as e:
            logger.error(f"Error in comprehensive blueprint analysis: {e}")
            blueprint_analysis['error'] = str(e)
        
        return blueprint_analysis
    
    def _determine_device_investigation_needs(self, blueprint_data, issue_description):
        """Determine if device commands are needed based on blueprint analysis and issue"""
        issue_lower = issue_description.lower()
        
        # Check if blueprint data is sufficient
        blueprint_errors = blueprint_data.get('errors', {})
        error_count = 0
        if isinstance(blueprint_errors, dict):
            for error_type, errors in blueprint_errors.items():
                if isinstance(errors, list):
                    error_count += len(errors)
        
        # Issues that require device commands (operational state not in blueprint)
        operational_issues = [
            'route', 'routing', 'reachability', 'ping', 'connectivity',
            'bgp session', 'ospf', 'isis', 'convergence', 'performance',
            'latency', 'packet loss', 'evpn routes', 'mac table'
        ]
        
        # Issues that blueprint can usually handle
        configuration_issues = [
            'config', 'vlan', 'interface', 'policy', 'design', 'intent'
        ]
        
        needs_device_investigation = False
        reason = "Blueprint data sufficient for configuration and design issues"
        commands = []
        devices = []
        
        # If blueprint shows errors, we might need device verification
        if error_count > 0:
            needs_device_investigation = True
            reason = f"Blueprint shows {error_count} errors requiring device verification"
            
            # Get spine devices for verification
            nodes_data = blueprint_data.get('nodes', {})
            devices = self._extract_spine_devices({'nodes': nodes_data.get('nodes', {})})
            
            commands = [
                {
                    'command': 'show system alarms',
                    'reason': 'Check for hardware/system issues causing blueprint errors',
                    'max_devices': 2
                }
            ]
        
        # If issue description suggests operational problems
        if any(issue_keyword in issue_lower for issue_keyword in operational_issues):
            needs_device_investigation = True
            reason = f"Issue '{issue_description}' requires operational state verification"
            
            if not devices:  # Get devices if we don't have them
                nodes_data = blueprint_data.get('nodes', {})
                devices = self._extract_spine_devices({'nodes': nodes_data.get('nodes', {})})
            
            # Add operational commands based on issue type
            if 'route' in issue_lower or 'evpn' in issue_lower:
                commands.append({
                    'command': 'show route table bgp.evpn.0 summary',
                    'reason': 'Check current EVPN route state',
                    'max_devices': 2
                })
            
            if 'bgp' in issue_lower:
                commands.append({
                    'command': 'show bgp summary',
                    'reason': 'Check BGP session operational state',
                    'max_devices': 2
                })
        
        return {
            'needed': needs_device_investigation,
            'reason': reason,
            'commands': commands,
            'devices': devices
        }


def main():
    """Main entry point"""
    if len(sys.argv) != 2:
        print("Usage: python mcp_server.py <openapi_spec_path>")
        sys.exit(1)
    
    openapi_spec_path = sys.argv[1]
    server = OpenAPIMCPServer()
    server.run(openapi_spec_path)


if __name__ == "__main__":
    main()

# add graph quirers and pdf docs?
# Add user docs
# add dynamic command discovery
# add adhd stuff