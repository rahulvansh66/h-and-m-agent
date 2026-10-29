variable "region" {
  description = "AWS region (AgentCore must be available there)."
  type        = string
  default     = "us-east-1"
}

variable "aws_profile" {
  description = "Named AWS CLI profile Terraform runs as (null uses the default credential chain)."
  type        = string
  default     = null
}

variable "name_prefix" {
  description = "Prefix for every resource name; lower-case letters, digits and hyphens."
  type        = string
  default     = "hm-demo"
}

variable "code_zip_path" {
  description = "Local path of the deployment zip (entry-point files + linux/arm64 dependencies) built by the notebook."
  type        = string
}

variable "neon_connection_string" {
  description = "Neon Postgres connection string for the order MCP server (set via TF_VAR_neon_connection_string). Ends up in the Runtime environment and in the state file: demo level only, use Secrets Manager in production."
  type        = string
  sensitive   = true
}

variable "groq_api_key" {
  description = "Groq API key for the supervisor's LLM calls (set via TF_VAR_groq_api_key). Empty = the supervisor falls back to keyword rules. Same demo-level secret handling as the Neon string."
  type        = string
  sensitive   = true
  default     = ""
}

variable "groq_model" {
  description = "Groq model name (role \"small\" in config/base.yaml)."
  type        = string
  default     = "openai/gpt-oss-20b"
}

variable "python_runtime" {
  description = "Python version of the Runtime code environment."
  type        = string
  default     = "PYTHON_3_12"
}
