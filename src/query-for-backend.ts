import * as dotenv from "dotenv";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { Pinecone } from "@pinecone-database/pinecone";
import { Content, GoogleGenAI } from "@google/genai";

dotenv.config();

// Initialize LLM
const ai = new GoogleGenAI({});
const History: Content[] = [];

// Initialize Pinecone
const pinecone = new Pinecone();
const pineconeIndex = pinecone.Index(process.env.PINECONE_INDEX_NAME!);

// Configure Embedding Model
const embeddings = new GoogleGenerativeAIEmbeddings({
  apiKey: process.env.GEMINI_API_KEY,
  model: "text-embedding-004",
});

async function transformQuery(question: string) {
  History.push({
    role: "user",
    parts: [{ text: question }],
  });

  const response = await ai.models.generateContent({
    model: "gemini-2.0-flash",
    contents: History,
    config: {
      systemInstruction: `You are a query rewriting expert. Based on the provided chat history, rephrase the "Follow Up user Question" into a complete, standalone question that can be understood without the chat history.
      Only output the rewritten question and nothing else.
        `,
    },
  });

  History.pop();

  return response.text;
}

export const chatting = async (question: string) => {
  if (!question) return;

  // Get detailed meaning of the question
  const queries = await transformQuery(question);
  if (!queries) return;

  // Convert question into vector
  const queryVector = await embeddings.embedQuery(queries);

  // Search similar vectors in DB
  const searchResults = await pineconeIndex.query({
    topK: 10,
    vector: queryVector,
    includeMetadata: true,
  });

  // Get the relevant text
  const context = searchResults.matches
    .map((match) => match.metadata?.text)
    .join("\n\n---\n\n");

  // Send Query + Context to LLM
  History.push({
    role: "user",
    parts: [{ text: queries }],
  });

  const response = await ai.models.generateContent({
    model: "gemini-2.0-flash",
    contents: History,
    config: {
      systemInstruction: `You have to behave like a Data Structure and Algorithm Expert.
      You will be given a context of relevant information and a user question.
      Your task is to answer the user's question based ONLY on the provided context.
      If the answer is not in the context, you must say "I could not find the answer in the provided document."
      Keep your answers clear, concise, and educational.
        
        Context: ${context}
        `,
    },
  });

  History.push({
    role: "model",
    parts: [{ text: response.text }],
  });
};
