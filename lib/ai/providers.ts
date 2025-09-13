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
            // UI label: GPT-4.1
            'chat-model': openai('gpt-4.1'),
            // Keep titles/artifacts on a capable default; align with GPT-4.1
            'title-model': openai('gpt-4.1'),
            'artifact-model': openai('gpt-4.1'),
          };
        })(),
      },
    });
