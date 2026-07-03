# AWS ECS Deployment Notes

1. Build the Docker image locally.
2. Push it to Amazon ECR.
3. Create an ECS Fargate task definition.
4. Expose port `8000` through an Application Load Balancer.
5. Configure environment variables from `.env.example`.
6. Use `/health` as the load-balancer health-check route.

This file is intentionally a deployment checklist, because AWS account IDs, ECR repository names, IAM roles, and VPC settings are account-specific.
