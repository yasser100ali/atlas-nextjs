export const DEFAULT_CHAT_MODEL: string = 'chat-model';

export interface ChatModel {
  id: string;
  name: string;
  description: string;
}

export const chatModels: Array<ChatModel> = [
  {
    id: 'chat-model',
    name: 'GPT-5',
    description:
      'Flagship OpenAI multimodal model for high‑quality text and vision tasks.',
  },
  {
    id: 'chat-model-reasoning',
    name: 'Deep-Research',
    description:
      'Optimized for long, step‑by‑step reasoning and research‑style analysis.',
  },
];
