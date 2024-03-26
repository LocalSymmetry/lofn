# Lofn AI Art Generator

Lofn is an open-source AI art generator that utilizes advanced natural language processing and image generation techniques to create unique and compelling artwork. It stands out from other AI art generators through its innovative use of Tree of Thoughts prompting and a critic/artist refinement methodology.

## Table of Contents
- [Installation](#installation)
  - [Docker](#docker)
- [Usage](#usage)
  - [DALL-E 3 Integration](#dall-e-3-integration)
  - [Discord Integration](#discord-integration)
- [What Makes Lofn Unique](#what-makes-lofn-unique)
  - [Tree of Thoughts Prompting](#tree-of-thoughts-prompting)
  - [Critic/Artist Refinement Methodology](#criticartist-refinement-methodology)
  - [Backend Infrastructure](#backend-infrastructure)
- [Awards and Recognition](#awards-and-recognition)
- [Comparative Examples](#comparative-examples)
- [Contributing](#contributing)
- [License](#license)

## Installation

To set up Lofn, follow these steps:

1. Clone the Lofn repository:
```bash
git clone https://github.com/yourusername/lofn.git
```

2. Navigate to the project directory:
```bash
cd lofn
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

4. Set up your API keys for the necessary services (e.g., OpenAI, Anthropic) in the `config.yaml` file.

5. Run the Lofn UI:
```bash
streamlit run lofn_ui.py
```

### Docker

Alternatively, you can use Docker to run Lofn:

1. Build the Docker image:
```bash
docker build -t lofn .
```

2. Run the Docker container:
```bash
docker run -p 8501:8501 lofn
```

3. Access the Lofn UI by opening `http://localhost:8501` in your web browser.

## Usage

1. Open the Lofn UI in your web browser.

2. Enter your idea or concept in the text area provided.

3. Adjust the settings in the sidebar, such as the maximum number of retries, the model to use, the temperature, and the competition mode.

4. Click the "Generate Concepts" button to generate concept and medium pairings.

5. Select the desired concept and medium pairs from the generated list.

6. Click the "Generate MJ Prompts" button to generate Midjourney prompts for the selected pairs.

7. Review the generated prompts and make any necessary adjustments.

8. Use the generated prompts with the Midjourney AI to create your artwork.

### DALL-E 3 Integration

Lofn also generates prompts that can be used with DALL-E 3. After generating the Midjourney prompts, you will find a section with the DALL-E 3 prompt. Copy this prompt and paste it into ChatGPT or any other platform that supports DALL-E 3 to generate your artwork.

### Discord Integration

Lofn can send the generated prompts directly to a Discord channel using webhooks. To set up Discord integration:

1. Create a webhook in your Discord server settings.

2. Copy the webhook URL.

3. Paste the webhook URL in the "Discord Webhook URL" field in the Lofn UI sidebar.

4. Enable the "Send to Discord" option.

Now, when you generate prompts, they will be automatically sent to your specified Discord channel.

## What Makes Lofn Unique

### Tree of Thoughts Prompting

Lofn utilizes a Tree of Thoughts (ToT) prompting approach, which enables the AI to explore and generate ideas in a more strategic and coherent manner. By maintaining a tree of thoughts, where each thought represents an intermediate step towards solving the problem, Lofn can self-evaluate its progress and make informed decisions during the art generation process.

### Critic/Artist Refinement Methodology

Lofn incorporates a critic/artist refinement methodology to iteratively improve the generated artwork. The AI takes on the roles of both the critic and the artist, critiquing the generated artwork based on predefined criteria and then refining it based on the feedback. This process allows Lofn to create more sophisticated and polished artwork compared to other AI art generators.

### Backend Infrastructure

Lofn supports both OpenAI and Anthropic's Claude as backend infrastructure for natural language processing. This flexibility allows users to choose their preferred AI model based on their requirements and available resources.

## Awards and Recognition

Coming Soon

## Comparative Examples

Here are a few comparative examples showcasing the differences between Lofn and other AI art generators:

1. **Consistency**: Lofn's Tree of Thoughts prompting ensures that the generated artwork maintains a consistent theme and style throughout the creation process. In contrast, other AI art generators may produce artwork with inconsistent or unrelated elements.

2. **Refinement**: Lofn's critic/artist refinement methodology allows for iterative improvements to the generated artwork. Other AI art generators often produce a single output without the ability to refine or enhance it further.

3. **Customization**: Lofn provides users with more control over the art generation process through its extensive settings and options. Users can fine-tune the generation process to align with their specific preferences and requirements.

4. **Originality**: Lofn's unique combination of Tree of Thoughts prompting and critic/artist refinement results in more original and creative artwork compared to other AI art generators that rely on predefined styles or templates.

## Contributing

We welcome contributions to Lofn! If you'd like to contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes and commit them with descriptive commit messages.
4. Push your changes to your forked repository.
5. Submit a pull request to the main Lofn repository.

Please ensure that your contributions adhere to our code of conduct and guidelines.

## License

Lofn is released under the [Apache 2.0 License](LICENSE).
