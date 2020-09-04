
output "this_apigatewayv2_api_api_endpoint" {
    description = "PyGrid network's API endpoint"
    value = module.api_gateway.this_apigatewayv2_api_api_endpoint
}

output "this_rds_cluster_arn" {
    description = "RDS Cluster ARN"
    value = module.aurora.this_rds_cluster_arn
}