
# MCP documentation for Tongyi-MAI/Z-Image-Turbo
MCP Tools: 1



This page documents two transports: Streamable HTTP and STDIO.

### Streamable HTTP

MCP Server URL (Streamable HTTP): https://tongyi-mai-z-image-turbo.hf.space/gradio_api/mcp/

1 available MCP tools, resources, and prompts: 

### Z_Image_Turbo_generate
Type: tool
Description: Generate an image using the Z-Image model based on the provided prompt and settings. This function is triggered when the user clicks the "Generate" button. It processes the input prompt (optionally enhancing it), configures generation parameters, and produces an image using the Z-Image diffusion transformer pipeline. If a user passes a file as an input, use the upload_file_to_gradio tool, if present, to upload the file to the gradio app and create a Gradio File Input. Then use the returned path as the input to the tool. Returns: tuple: (gallery_images, seed_str, seed_int), - gallery_images: Updated list of generated images including the new image, - seed_str: String representation of the seed used for generation, - seed_int: Integer representation of the seed used for generation
Parameters: 7
- prompt (string): Text prompt describing the desired image content
- resolution (string): Output resolution in format "WIDTHxHEIGHT ( RATIO )" (e.g., "1024x1024 ( 1:1 )")
- seed (integer): Seed for reproducible generation
- steps (number): Number of inference steps for the diffusion process
- shift (number): Time shift parameter for the flow matching scheduler
- random_seed (boolean): Whether to generate a new random seed, if True will ignore the seed input
- gallery_images (array): List of previously generated images to append to (only needed for the Gradio UI)


Streamable HTTP Transport: To add this MCP to clients that support Streamable HTTP, simply add the following configuration to your MCP config.

```json
{
  "mcpServers": {
    "gradio": {
      "url": "https://tongyi-mai-z-image-turbo.hf.space/gradio_api/mcp/"
    },
    "upload_files_to_gradio": {
      "command": "uvx",
      "args": [
        "--from",
        "gradio[mcp]",
        "gradio",
        "upload-mcp",
        "https://tongyi-mai-z-image-turbo.hf.space/",
        "<UPLOAD_DIRECTORY>"
      ]
    }
  }
}
```

The `upload_files_to_gradio` tool uploads files from your local `UPLOAD_DIRECTORY` (or any of its subdirectories) to the Gradio app. 
This is needed because MCP servers require files to be provided as URLs. You can omit this tool if you prefer to upload files manually. This tool requires [uv](https://docs.astral.sh/uv/getting-started/installation/) to be installed.

### STDIO Transport


1 available MCP tools, resources, and prompts: 

### Z_Image_Turbo_generate
Type: tool
Description: Generate an image using the Z-Image model based on the provided prompt and settings. This function is triggered when the user clicks the "Generate" button. It processes the input prompt (optionally enhancing it), configures generation parameters, and produces an image using the Z-Image diffusion transformer pipeline. If a user passes a file as an input, use the upload_file_to_gradio tool, if present, to upload the file to the gradio app and create a Gradio File Input. Then use the returned path as the input to the tool. Returns: tuple: (gallery_images, seed_str, seed_int), - gallery_images: Updated list of generated images including the new image, - seed_str: String representation of the seed used for generation, - seed_int: Integer representation of the seed used for generation
Parameters: 7
- prompt (string): Text prompt describing the desired image content
- resolution (string): Output resolution in format "WIDTHxHEIGHT ( RATIO )" (e.g., "1024x1024 ( 1:1 )")
- seed (integer): Seed for reproducible generation
- steps (number): Number of inference steps for the diffusion process
- shift (number): Time shift parameter for the flow matching scheduler
- random_seed (boolean): Whether to generate a new random seed, if True will ignore the seed input
- gallery_images (array): List of previously generated images to append to (only needed for the Gradio UI)


STDIO Transport: For clients that only support stdio (e.g. Claude Desktop), first [install Node.js](https://nodejs.org/en/download/). Then, you can use the following command:

```json
{
  "mcpServers": {
    "gradio": {
      "command": "npx",
      "args": [
        "mcp-remote",
        "https://tongyi-mai-z-image-turbo.hf.space/gradio_api/mcp/",
        "--transport",
        "streamable-http"
      ]
    },
    "upload_files_to_gradio": {
      "command": "uvx",
      "args": [
        "--from",
        "gradio[mcp]",
        "gradio",
        "upload-mcp",
        "https://tongyi-mai-z-image-turbo.hf.space/",
        "<UPLOAD_DIRECTORY>"
      ]
    }
  }
}
```

The `upload_files_to_gradio` tool uploads files from your local `UPLOAD_DIRECTORY` (or any of its subdirectories) to the Gradio app. 
This is needed because MCP servers require files to be provided as URLs. You can omit this tool if you prefer to upload files manually. This tool requires [uv](https://docs.astral.sh/uv/getting-started/installation/) to be installed.

Read more about the MCP in the [Gradio docs](https://www.gradio.app/guides/building-mcp-server-with-gradio).


