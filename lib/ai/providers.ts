import { customProvider } from 'ai';
import { createOpenAI } from '@ai-sdk/openai';
import { isTestEnvironment } from '../constants';

export const myProvider = isTestEnvironment
  ? (() => {
      const {
        artifactModel,
        chatModel,
        reasoningModel,
        titleModel,
      } = require('./models.mock');
      return customProvider({
        languageModels: {
          'chat-model': chatModel,
          'chat-model-reasoning': reasoningModel,
          'title-model': titleModel,
          'artifact-model': artifactModel,
        },
      });
    })()
  : customProvider({
      languageModels: {
        // Create OpenAI provider using env var
        ...(() => {
          const openai = createOpenAI({
            apiKey: process.env.OPENAI_API_KEY,
            headers: {
              'OpenAI-Beta': 'assistants=v2',
            },
          });
          return {
            // UI label: GPT-5
            'chat-model': openai('gpt-5'),
            // UI label: Deep-Research
            'chat-model-reasoning': openai('o4-mini-deep-research'),
            // Keep titles/artifacts on a capable default; align with GPT-5
            'title-model': openai('gpt-5'),
            'artifact-model': openai('gpt-5'),
          };
        })(),
      },
    });
