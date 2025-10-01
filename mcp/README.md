# FastMCP Server Setup for Claude Desktop

This guide will help you install, configure, and connect a FastMCP-enabled server to Claude Desktop using the `uv` package manager.

***

## Prerequisites

- **Claude Desktop** installed
- **Python** (3.8+)
- **uv** (Python package manager) installed and in your system PATH
- **VS Code** (recommended, but any code editor will work)

***

## Step 1: Install `uv`

Install `uv` globally so that Claude Desktop and your development environment can use it:

```bash
pip install uv
```


***

## Step 2: Create and Initialize Your Project

- Create a new project folder:

```bash
mkdir fastmcp_demo
cd fastmcp_demo
```

- Open this folder in VS Code:

```bash
code .
```

- Initialize the project with `uv`:

```bash
uv init
```


***

## Step 3: Add FastMCP Dependency

Add FastMCP to your project’s environment:

```bash
uv add fastmcp
```


***

## Step 4: Create a Basic Server

- Create `main.py` (example FastMCP tool):

```python
from fastmcp import FastMCP

mcp = FastMCP(name="Demo Server")

@mcp.tool()
def hello(name: str = "world"):
    return f"Hello, {name}!"

if __name__ == "__main__":
    mcp.run()
```


***

## Step 5: Test the Server

- Start the server in development mode:

```bash
uv run fastmcp dev main.py
```

Confirm there are no errors and FastMCP starts up correctly.
- To run in production/server mode:

```bash
uv run fastmcp run main.py
```


***

## Step 6: Add the Server to Claude Desktop

- Use the FastMCP CLI helper to register with Claude Desktop:

```bash
uv run fastmcp install claude-desktop main.py
```

This will guide you to set up and confirm the server integration.

***

## Step 7: Edit Claude Desktop Config

- Open the `claude_desktop_config.json` config file for Claude Desktop.
    - **Windows:** `%APPDATA%\Claude\claude_desktop_config.json`
- Add (or verify) the following configuration under `mcpServers`:

```json
{
  "mcpServers": {
    "Demo Server": {
      "command": "C:\\Users\\abhi1\\anaconda3\\Scripts\\uv.exe",
      "args": [
        "run",
        "--with",
        "fastmcp",
        "fastmcp",
        "run",
        "D:\\Practical\\mcp_tutorial\\main.py"
      ],
      "env": {},
      "transport": "stdio",
      "type": null,
      "cwd": null,
      "timeout": null,
      "description": null,
      "icon": null,
      "authentication": null
    }
  }
}
```


***

## Step 8: Restart Claude Desktop

- Close and reopen Claude Desktop.
- Your "Demo Server" will appear as a tool in Claude’s interface.
- Test it by invoking `hello` or another tool you defined.

***

## Tips and Troubleshooting

- Be sure `uv` is installed globally and available in the PATH Claude uses.
- Double-check file paths and the Python executable location in your config.
- Use the Claude Desktop logs for debugging integration issues.

***
