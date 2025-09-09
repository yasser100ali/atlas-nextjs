import { Agent, OpenAIResponsesModel, webSearchTool } from '@openai/agents';
import OpenAI from 'openai';
import { z } from 'zod';

const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY || '' });

const Finding = z.object({
  title: z.string(),
  // OpenAI Responses API rejects JSON Schema format "uri"; zod.url() maps to that.
  // Accept plain string to satisfy the validator.
  url: z.string(),
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

export const researchAgent = new Agent({
  name: 'Research_Assistant',
  model: new OpenAIResponsesModel(client, 'o4-mini-deep-research'),
  instructions: `
    You are a Deep Research agent.

WORKFLOW
1) Interpret the user's query (and optional focus area).
2) Use your built-in browsing & analysis to gather facts, stats, and differing viewpoints from reputable, recent sources.
3) Build a list named findings with dicts: title, url, snippet.
4) Output a final JSON object EXACTLY ONCE, then STOP. The JSON must match:
   {"success":true,"query":"...","focus_area":"...","answer":"...","key_insights":["..."],"sources":[{"title":"...","url":"...","snippet":"..."}],"status":"final"}

CONSTRAINTS
- Prefer authoritative and recent sources when recency matters.
- Keep the synthesis concise, neutral, and evidence-based.

END OF RESEARCH.
    `.trim(),
  tools: [webSearchTool()],
});
