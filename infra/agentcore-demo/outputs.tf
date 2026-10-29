output "region" {
  value = var.region
}

output "supervisor_runtime_arn" {
  description = "Invoke this one (bedrock-agentcore InvokeAgentRuntime)."
  value       = aws_bedrockagentcore_agent_runtime.supervisor.agent_runtime_arn
}

output "mcp_runtime_arn" {
  value = aws_bedrockagentcore_agent_runtime.mcp.agent_runtime_arn
}

output "shipment_runtime_arn" {
  value = aws_bedrockagentcore_agent_runtime.shipment.agent_runtime_arn
}

output "mcp_url" {
  description = "Order MCP server on Runtime (JWT protected)."
  value       = local.mcp_url
}

output "shipment_url" {
  description = "Shipment A2A agent on Runtime (SigV4 protected)."
  value       = local.shipment_url
}

output "gateway_url" {
  description = "MCP endpoint of the Gateway (already ends in /mcp)."
  value       = aws_bedrockagentcore_gateway.this.gateway_url
}

output "cognito_token_endpoint" {
  value = "https://${aws_cognito_user_pool_domain.m2m.domain}.auth.${var.region}.amazoncognito.com/oauth2/token"
}

output "cognito_client_id" {
  value = aws_cognito_user_pool_client.gateway.id
}

output "cognito_client_secret" {
  value     = aws_cognito_user_pool_client.gateway.client_secret
  sensitive = true
}
