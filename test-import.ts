// test-import.ts
import { Agent } from "@openai/agents";
import { webSearchTool } from "@openai/agents-openai";
import { z } from "zod";

console.log("OK", Agent, webSearchTool, z.string().parse("hi"));