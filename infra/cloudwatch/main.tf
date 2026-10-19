# CloudWatch Logs for the H&M assistant: one log group per environment plus a
# least-privilege IAM policy for writing to it. Pay-per-use only; no always-on resources.

terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region  = var.region
  profile = var.aws_profile
}

variable "region" {
  description = "AWS region; must match logging.cloudwatch.region in config/."
  type        = string
  default     = "us-east-1"
}

variable "aws_profile" {
  description = "Named AWS CLI profile Terraform runs as (null uses the default credential chain)."
  type        = string
  default     = null
}

variable "environments" {
  description = "Environments to create log groups for (match config/<env>.yaml)."
  type        = set(string)
  default     = ["dev"]
}

variable "retention_days" {
  description = "Days to keep logs; storage is cheap (~$0.03/GB-month, first 5 GB free)."
  type        = number
  default     = 90
}

resource "aws_cloudwatch_log_group" "app" {
  for_each          = var.environments
  name              = "/hm-agent/${each.key}"
  retention_in_days = var.retention_days
}

# Only what the watchtower handler needs: create streams and put events in these groups.
data "aws_iam_policy_document" "write_logs" {
  statement {
    actions   = ["logs:CreateLogStream", "logs:PutLogEvents", "logs:DescribeLogStreams"]
    resources = flatten([for g in aws_cloudwatch_log_group.app : [g.arn, "${g.arn}:log-stream:*"]])
  }
}

resource "aws_iam_policy" "write_logs" {
  name        = "hm-agent-write-logs"
  description = "Allow the H&M assistant to write to its CloudWatch log groups"
  policy      = data.aws_iam_policy_document.write_logs.json
}

output "log_group_names" {
  description = "Set logging.cloudwatch.log_group in config/ to the matching name."
  value       = [for g in aws_cloudwatch_log_group.app : g.name]
}

output "write_logs_policy_arn" {
  description = "Attach to the IAM user/role behind CLOUDWATCH_AWS_PROFILE."
  value       = aws_iam_policy.write_logs.arn
}
