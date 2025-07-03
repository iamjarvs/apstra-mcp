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
            
            # Start the server (authentication will happen on first API call)
            logger.info("Starting OpenAPI MCP Server (authentication will happen on first API call)")
            self.mcp.run()
            
        except Exception as e:
            logger.error(f"Server error: {e}")
            raise


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