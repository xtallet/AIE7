# Setting Up Your Open-Source Endpoint

> NOTE: If you do not wish to purchase $50 of compute credits for T2 (for using dedicated endpoints) you can instead skip the following set-up and simply use the serverless endpoints offered at: 

- `openai/gpt-oss-20b`

## Your Generator

First, you'll want to navigate to [api.together.ai/models](https://api.together.ai/models), and search for the model we'll be using today: 

- `gpt-oss`

We're going to select the OpenAI GPT-OSS 20B model by clicking on it:

![image](./images/Z82ArVL.png)

Next, we're going to click on "Create Dedicated Endpoint" to spin up a dedicated endpoint. 

![image](./images/dWqtZ6i.png)

You'll want to set your settings as follows and then click "Deploy": 

![image](./images/eZvZGZo%20-%20Imgur.png)

> NOTE: Please ensure you have an Auto-shutdown selected - a value like `1 hour` is useful to ensure your endpoint does not spin down during class.

After you click "Deploy" - you should see the endpoint spinning up, as well as a name for your new endpoint!

> NOTE: You'll want to make sure you get an API key from together.ai as well! You can follow the instructions [here](https://docs.together.ai/reference/authentication-1)

## Your Embeddings 

Together offers serverless endpoints for embedding models, we'll be using the [BAAI-BGE-Large-1.5](https://huggingface.co/BAAI/bge-large-en-v1.5) model today!

- `BAAI/bge-large-en-v1.5`

### ❓ Question #1: 

What is the difference between serverless and dedicated endpoints?

##### ✅ Answer:

Serverless endpoints and dedicated endpoints are two different ways of accessing a model in the cloud:

- <b>Serverless endpoints</b><br>
Work “on demand”: they are only activated when you make a request.<br>
You don’t need to configure or maintain servers.<br>
They automatically scale based on demand.<br>
The cost depends on usage (for example, processed tokens or compute time).<br>
They are ideal for variable workloads or experimentation.<br>

- <b>Dedicated endpoints</b><br>
You reserve dedicated compute instances for your application.<br>
Performance is more predictable (because the resources are yours, not shared).<br>
They provide lower latency and greater control over availability.<br>
The cost is fixed (you pay for the instances while they are running, even if unused).<br>
They are more convenient when you need constant usage, low latency, or strict SLA compliance.<br>

<b>Key difference :</b><br>

<b>Serverless</b> = flexible, pay only for what you use, great for intermittent workloads.<br>
<b>Dedicated</b> = exclusive and always-available resources, better for production with high traffic or critical needs.