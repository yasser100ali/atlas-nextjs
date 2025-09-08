import { customProvider, wrapLanguageModel } from 'ai';
import OpenAI from '@ai-sdk/openai';
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
        // Main chat model (gpt-4o-mini is a good fast default; adjust as needed)
        'chat-model': OpenAI('gpt-4o-mini'),

        // Reasoning model (use o3-mini for lightweight reasoning traces)
        'chat-model-reasoning': wrapLanguageModel({
          model: OpenAI('o3-mini'),
        }),

        // Title generation and artifacts can use a cost-effective GPT model
        'title-model': OpenAI('gpt-4o-mini'),
        'artifact-model': OpenAI('gpt-4o-mini'),
      },
    });
