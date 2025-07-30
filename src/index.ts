import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { Document } from "@langchain/core/documents";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import * as dotenv from "dotenv";
import { Pinecone } from "@pinecone-database/pinecone";
import { PineconeStore } from "@langchain/pinecone";

dotenv.config();

// Configure Embedding Model
const embeddings = new GoogleGenerativeAIEmbeddings({
  apiKey: process.env.GEMINI_API_KEY,
  model: "text-embedding-004",
});

// Initialize PineCone
const pinecone = new Pinecone();
const pineconeIndex = pinecone.Index(process.env.PINECONE_INDEX_NAME!);

const indexDoc = async () => {
  const PDF_PATH = "./src/dsa.pdf";
  const pdfLoader = new PDFLoader(PDF_PATH);
  const rawDocs: Document<Record<string, any>>[] = await pdfLoader.load();

  chunkifyDoc(rawDocs);
};

const chunkifyDoc = async (rawDocs: Document<Record<string, any>>[]) => {
  const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 200,
  });
  const chunkedDocs: Document[] = await textSplitter.splitDocuments(rawDocs);
  putIntoVectorDB(chunkedDocs);
};

const putIntoVectorDB = async (chunkedDocs: Document[]) => {
  await PineconeStore.fromDocuments(chunkedDocs, embeddings, {
    pineconeIndex,
    maxConcurrency: 5, // Free tier thingss
  });
};

indexDoc();
