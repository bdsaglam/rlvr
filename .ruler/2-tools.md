# Tools 

There are some tools available to you.

## MCP (Model Context Protocol)
- `context7`: for reading documentation for unfamiliar tools
- `hf-mcp-server`: for finding resources (models, datasets, papers, etc.) on Hugging Face
- And other MCP servers configured for this workspace

## Just commands
- `just` manages workspace commands (already installed)
- Global commands via `gust` (alias for `just -g`):
  - `gust transcribe {filepath}` - Convert media to text
  - `gust pdf2md {filepath}` - Convert PDF to markdown
- Add project-specific commands to local `justfile`