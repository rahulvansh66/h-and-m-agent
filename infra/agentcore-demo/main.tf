# AgentCore demo stack: an MCP server and an A2A agent on AgentCore Runtime (code zip in S3, no Docker),
# a Gateway in front of the MCP server, and the supervisor Runtime that calls both.
#
#   supervisor (Runtime, HTTP)  --SigV4-->  Gateway (IAM inbound)  --OAuth token-->  order MCP server (Runtime, MCP, JWT)
#                               --SigV4-->  shipment agent (Runtime, A2A, IAM)
#
# Everything is pay-per-use (Runtime bills active CPU/memory seconds, Gateway bills per call, Cognito and S3 are
# near zero at demo volume). `terraform destroy` removes all of it.

terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.0" # AgentCore resources exist from provider 6.x
    }
  }
}

provider "aws" {
  region  = var.region
  profile = var.aws_profile
}

data "aws_caller_identity" "current" {}

locals {
  account_id = data.aws_caller_identity.current.account_id
  # Runtime names allow letters, digits and underscores only.
  rt_prefix           = replace(var.name_prefix, "-", "_")
  runtime_invoke_base = "https://bedrock-agentcore.${var.region}.amazonaws.com/runtimes"
  cognito_issuer      = "https://cognito-idp.${var.region}.amazonaws.com/${aws_cognito_user_pool.m2m.id}"
  agentcore_arn       = "arn:aws:bedrock-agentcore:${var.region}:${local.account_id}"
  # Address of the order MCP server on Runtime (what the Gateway target points at).
  mcp_url = "${local.runtime_invoke_base}/${urlencode(aws_bedrockagentcore_agent_runtime.mcp.agent_runtime_arn)}/invocations?qualifier=DEFAULT"
  # Address of the shipment A2A agent on Runtime (what the supervisor calls). A2A is mounted at the path root.
  shipment_url = "${local.runtime_invoke_base}/${urlencode(aws_bedrockagentcore_agent_runtime.shipment.agent_runtime_arn)}/invocations/"
}

# ---- code package ----------------------------------------------------------------------------------
# One zip holds the three entry-point files plus linux/arm64 dependencies; each Runtime picks its entry point.

resource "aws_s3_bucket" "code" {
  bucket        = "${var.name_prefix}-code-${local.account_id}-${var.region}"
  force_destroy = true
}

resource "aws_s3_bucket_public_access_block" "code" {
  bucket                  = aws_s3_bucket.code.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_versioning" "code" {
  bucket = aws_s3_bucket.code.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_object" "code" {
  bucket     = aws_s3_bucket.code.id
  key        = "agent-code/app.zip"
  source     = var.code_zip_path
  etag       = filemd5(var.code_zip_path)
  depends_on = [aws_s3_bucket_versioning.code]
}

# ---- IAM -------------------------------------------------------------------------------------------

data "aws_iam_policy_document" "agentcore_trust" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["bedrock-agentcore.amazonaws.com"]
    }
    condition {
      test     = "StringEquals"
      variable = "aws:SourceAccount"
      values   = [local.account_id]
    }
    condition {
      test     = "ArnLike"
      variable = "aws:SourceArn"
      values   = ["${local.agentcore_arn}:*"]
    }
  }
}

# Permissions every Runtime needs: write logs, read its own code zip, get a workload identity token.
data "aws_iam_policy_document" "runtime_base" {
  statement {
    actions   = ["logs:CreateLogGroup", "logs:DescribeLogStreams", "logs:CreateLogStream", "logs:PutLogEvents"]
    resources = ["arn:aws:logs:${var.region}:${local.account_id}:log-group:/aws/bedrock-agentcore/runtimes/*"]
  }
  statement {
    actions   = ["logs:DescribeLogGroups"]
    resources = ["arn:aws:logs:${var.region}:${local.account_id}:log-group:*"]
  }
  statement {
    actions   = ["s3:GetObject", "s3:GetObjectVersion"]
    resources = ["${aws_s3_bucket.code.arn}/*"]
  }
  statement {
    actions = ["bedrock-agentcore:GetWorkloadAccessToken", "bedrock-agentcore:GetWorkloadAccessTokenForJWT", "bedrock-agentcore:GetWorkloadAccessTokenForUserId"]
    resources = [
      "${local.agentcore_arn}:workload-identity-directory/default",
      "${local.agentcore_arn}:workload-identity-directory/default/workload-identity/*",
    ]
  }
}

# MCP server and shipment agent: the base permissions only.
resource "aws_iam_role" "runtime_basic" {
  name               = "${var.name_prefix}-runtime-basic"
  assume_role_policy = data.aws_iam_policy_document.agentcore_trust.json
}

resource "aws_iam_role_policy" "runtime_basic" {
  role   = aws_iam_role.runtime_basic.id
  policy = data.aws_iam_policy_document.runtime_base.json
}

# Supervisor: base permissions plus calling the Gateway and the shipment agent.
data "aws_iam_policy_document" "supervisor_extra" {
  statement {
    actions   = ["bedrock-agentcore:InvokeGateway"]
    resources = [aws_bedrockagentcore_gateway.this.gateway_arn]
  }
  statement {
    actions = ["bedrock-agentcore:InvokeAgentRuntime", "bedrock-agentcore:GetAgentCard"]
    resources = [
      aws_bedrockagentcore_agent_runtime.shipment.agent_runtime_arn,
      "${aws_bedrockagentcore_agent_runtime.shipment.agent_runtime_arn}/runtime-endpoint/*",
    ]
  }
}

resource "aws_iam_role" "runtime_supervisor" {
  name               = "${var.name_prefix}-runtime-supervisor"
  assume_role_policy = data.aws_iam_policy_document.agentcore_trust.json
}

resource "aws_iam_role_policy" "runtime_supervisor_base" {
  name   = "base"
  role   = aws_iam_role.runtime_supervisor.id
  policy = data.aws_iam_policy_document.runtime_base.json
}

resource "aws_iam_role_policy" "runtime_supervisor_extra" {
  name   = "call-gateway-and-shipment"
  role   = aws_iam_role.runtime_supervisor.id
  policy = data.aws_iam_policy_document.supervisor_extra.json
}

# Gateway role: fetch the OAuth token (AgentCore Identity) it presents to the MCP server.
data "aws_iam_policy_document" "gateway" {
  statement {
    actions = ["bedrock-agentcore:GetResourceOauth2Token", "bedrock-agentcore:GetWorkloadAccessToken"]
    resources = [
      "${local.agentcore_arn}:token-vault/default",
      "${local.agentcore_arn}:token-vault/default/oauth2credentialprovider/*",
      "${local.agentcore_arn}:workload-identity-directory/default",
      "${local.agentcore_arn}:workload-identity-directory/default/workload-identity/*",
    ]
  }
  statement {
    actions   = ["secretsmanager:GetSecretValue"]
    resources = ["arn:aws:secretsmanager:${var.region}:${local.account_id}:secret:bedrock-agentcore-identity!default/oauth2/*"]
  }
}

resource "aws_iam_role" "gateway" {
  name               = "${var.name_prefix}-gateway"
  assume_role_policy = data.aws_iam_policy_document.agentcore_trust.json
}

resource "aws_iam_role_policy" "gateway" {
  role   = aws_iam_role.gateway.id
  policy = data.aws_iam_policy_document.gateway.json
}

# ---- Cognito: the Gateway proves itself to the MCP server with a client-credentials token ---------------

resource "aws_cognito_user_pool" "m2m" {
  name = "${var.name_prefix}-m2m"
}

resource "aws_cognito_user_pool_domain" "m2m" {
  domain       = "${var.name_prefix}-${local.account_id}"
  user_pool_id = aws_cognito_user_pool.m2m.id
}

resource "aws_cognito_resource_server" "mcp" {
  identifier   = "hm-mcp"
  name         = "H and M demo MCP"
  user_pool_id = aws_cognito_user_pool.m2m.id
  scope {
    scope_name        = "invoke"
    scope_description = "Call the order MCP server"
  }
}

resource "aws_cognito_user_pool_client" "gateway" {
  name                                 = "${var.name_prefix}-gateway"
  user_pool_id                         = aws_cognito_user_pool.m2m.id
  generate_secret                      = true
  allowed_oauth_flows                  = ["client_credentials"]
  allowed_oauth_flows_user_pool_client = true
  allowed_oauth_scopes                 = ["hm-mcp/invoke"]
  supported_identity_providers         = ["COGNITO"]
  depends_on                           = [aws_cognito_resource_server.mcp]
}

# AgentCore Identity stores the client secret and mints tokens for the Gateway.
resource "aws_bedrockagentcore_oauth2_credential_provider" "gateway_to_mcp" {
  name                       = "${var.name_prefix}-cognito"
  credential_provider_vendor = "CustomOauth2"
  oauth2_provider_config {
    custom_oauth2_provider_config {
      client_id     = aws_cognito_user_pool_client.gateway.id
      client_secret = aws_cognito_user_pool_client.gateway.client_secret
      oauth_discovery {
        discovery_url = "${local.cognito_issuer}/.well-known/openid-configuration"
      }
    }
  }
}

# ---- Runtimes --------------------------------------------------------------------------------------

resource "aws_bedrockagentcore_agent_runtime" "mcp" {
  agent_runtime_name = "${local.rt_prefix}_order_mcp"
  description        = "Order MCP server"
  role_arn           = aws_iam_role.runtime_basic.arn

  agent_runtime_artifact {
    code_configuration {
      entry_point = ["order_mcp_server.py"]
      runtime     = var.python_runtime
      code {
        s3 {
          bucket     = aws_s3_bucket.code.id
          prefix     = aws_s3_object.code.key
          version_id = aws_s3_object.code.version_id
        }
      }
    }
  }
  network_configuration {
    network_mode = "PUBLIC"
  }
  protocol_configuration {
    server_protocol = "MCP"
  }
  # Only callers holding a token from our Cognito client may call the server (that is the Gateway).
  authorizer_configuration {
    custom_jwt_authorizer {
      discovery_url   = "${local.cognito_issuer}/.well-known/openid-configuration"
      allowed_clients = [aws_cognito_user_pool_client.gateway.id]
    }
  }
  environment_variables = {
    HOST                            = "0.0.0.0"
    PORT                            = "8000"
    STATELESS_HTTP                  = "true"
    NEON_POSTGRES_CONNECTION_STRING = var.neon_connection_string
  }
}

resource "aws_bedrockagentcore_agent_runtime" "shipment" {
  agent_runtime_name = "${local.rt_prefix}_shipment_a2a"
  description        = "Shipment tracking A2A agent"
  role_arn           = aws_iam_role.runtime_basic.arn

  agent_runtime_artifact {
    code_configuration {
      entry_point = ["shipment_a2a_server.py"]
      runtime     = var.python_runtime
      code {
        s3 {
          bucket     = aws_s3_bucket.code.id
          prefix     = aws_s3_object.code.key
          version_id = aws_s3_object.code.version_id
        }
      }
    }
  }
  network_configuration {
    network_mode = "PUBLIC"
  }
  protocol_configuration {
    server_protocol = "A2A"
  }
  # No authorizer block = IAM (SigV4), the default.
  environment_variables = {
    HOST = "0.0.0.0"
    PORT = "9000"
  }
}

resource "aws_bedrockagentcore_agent_runtime" "supervisor" {
  agent_runtime_name = "${local.rt_prefix}_supervisor"
  description        = "Supervisor: routes to the order agent (MCP via Gateway) or the shipment agent (A2A)"
  role_arn           = aws_iam_role.runtime_supervisor.arn

  agent_runtime_artifact {
    code_configuration {
      entry_point = ["supervisor_runtime.py"]
      runtime     = var.python_runtime
      code {
        s3 {
          bucket     = aws_s3_bucket.code.id
          prefix     = aws_s3_object.code.key
          version_id = aws_s3_object.code.version_id
        }
      }
    }
  }
  network_configuration {
    network_mode = "PUBLIC"
  }
  protocol_configuration {
    server_protocol = "HTTP"
  }
  environment_variables = {
    HOST             = "0.0.0.0"
    PORT             = "8080"
    ORDER_MCP_URL    = aws_bedrockagentcore_gateway.this.gateway_url # already ends in /mcp
    ORDER_MCP_SIGV4  = "true"
    SHIPMENT_A2A_URL = local.shipment_url
    SHIPMENT_SIGV4   = "true"
    GROQ_API_KEY     = var.groq_api_key
    GROQ_MODEL       = var.groq_model
  }
}

# ---- Gateway ---------------------------------------------------------------------------------------

resource "aws_bedrockagentcore_gateway" "this" {
  name            = "${var.name_prefix}-gateway"
  description     = "One MCP endpoint in front of the order MCP server"
  role_arn        = aws_iam_role.gateway.arn
  protocol_type   = "MCP"
  authorizer_type = "AWS_IAM" # callers sign with SigV4; the supervisor's role is allowed InvokeGateway
}

resource "aws_bedrockagentcore_gateway_target" "order" {
  gateway_identifier = aws_bedrockagentcore_gateway.this.gateway_id
  name               = "order-management"
  description        = "Order MCP server hosted on AgentCore Runtime"

  target_configuration {
    mcp {
      mcp_server {
        endpoint = local.mcp_url
      }
    }
  }
  credential_provider_configuration {
    oauth {
      provider_arn = aws_bedrockagentcore_oauth2_credential_provider.gateway_to_mcp.credential_provider_arn
      scopes       = ["hm-mcp/invoke"]
      grant_type   = "CLIENT_CREDENTIALS" # machine-to-machine; stated explicitly because the provider otherwise reports it as an unexpected change
    }
  }
}
