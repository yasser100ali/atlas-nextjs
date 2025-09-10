import { Agent, tool, OpenAIResponsesModel, webSearchTool } from '@openai/agents';
import OpenAI from 'openai';
import { z } from 'zod';

const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY || '' });

const Finding = z.object({
  title: z.string(),
  url: z.string().url(),
  snippet: z.string().nullable(),
});

const SummarizeArgs = z.object({
  findings: z.array(Finding).min(1),
  query: z.string(),
  focus_area: z.string().nullable(),
  answer: z.string(),
  key_insights: z.array(z.string()).nullable(),
});

export type ResearchReport = z.infer<typeof SummarizeArgs> & {
  success: boolean;
  sources: z.infer<typeof Finding>[];
  status: 'final';
};

const summaryResearch = tool({
  name: 'summary_research',
  description: 'Return the final JSON research report.',
  parameters: SummarizeArgs,
  execute: async ({ findings, query, focus_area, answer, key_insights }) => ({
    success: true,
    query,
    focus_area: focus_area ?? '',
    answer,
    key_insights: key_insights ?? [],
    sources: findings,
    status: 'final' as const,
  }),
});

export const researchAgent = new Agent({
  name: 'Research_Assistant',
  model: new OpenAIResponsesModel(client, 'o4-mini-deep-research'),
  instructions: `
    You are a Deep Research agent.

WORKFLOW
1) Interpret the user's query (and optional focus area).
2) Use your built-in browsing & analysis to gather facts, stats, and differing viewpoints from reputable, recent sources.
3) Build a list named findings with dicts: title, url, snippet.
4) Call summary_research(...) EXACTLY ONCE, then STOP.

CONSTRAINTS
- Prefer authoritative and recent sources when recency matters.
- Keep the synthesis concise, neutral, and evidence-based.

END OF RESEARCH.
    `.trim(),
  tools: [webSearchTool(), summaryResearch],
  toolUseBehavior: { stopAtToolNames: ['summary_research'] },
});
