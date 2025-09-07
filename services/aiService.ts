
import { GoogleGenAI, Type } from "@google/genai";
import type { BillData, AiSettings, OllamaModel } from "../types";

// --- Common Schema and Utilities ---

const billSchema = {
  type: Type.OBJECT,
  properties: {
    accountName: { type: Type.STRING, description: "Account holder's full name, if available." },
    accountNumber: { type: Type.STRING, description: "The account number." },
    serviceAddress: { type: Type.STRING, description: "The full service address, if available." },
    statementDate: { type: Type.STRING, description: "The main date of the bill statement (e.g., 'October 5, 2017')." },
    servicePeriodStart: { type: Type.STRING, description: "The start date of the service period, if available (e.g., 'MM/DD/YYYY')." },
    servicePeriodEnd: { type: Type.STRING, description: "The end date of the service period, if available (e.g., 'MM/DD/YYYY')." },
    totalCurrentCharges: { type: Type.NUMBER, description: "The total amount due for the current period." },
    dueDate: { type: Type.STRING, description: "The payment due date, if available." },
    confidenceScore: { type: Type.NUMBER, description: "A score from 0.0 to 1.0 representing your confidence in the extracted data's accuracy. 1.0 is highest confidence." },
    confidenceReasoning: { type: Type.STRING, description: "A detailed explanation for the confidence score. Mention specific issues like blurriness, glare, unusual formatting, or hard-to-read sections. Be specific about which parts of the bill were problematic." },
    usageCharts: {
      type: Type.ARRAY, description: "An array of all usage charts found on the bill.",
      items: {
        type: Type.OBJECT, properties: {
          title: { type: Type.STRING, description: "The title of the chart." },
          unit: { type: Type.STRING, description: "The unit of measurement for the usage (e.g., kWh, Therms, m³)." },
          data: {
            type: Type.ARRAY, description: "The monthly data points from the chart.",
            items: {
              type: Type.OBJECT, properties: {
                month: { type: Type.STRING, description: "Abbreviated month name (e.g., Oct, Nov)." },
                usage: {
                    type: Type.ARRAY, description: "Usage values for each year shown in the chart.",
                    items: {
                        type: Type.OBJECT, properties: {
                            year: { type: Type.STRING, description: "The year of the usage value." },
                            value: { type: Type.NUMBER, description: "The numerical usage value for that year." }
                        }, required: ["year", "value"]
                    }
                },
              }, required: ["month", "usage"],
            },
          },
        }, required: ["title", "unit", "data"],
      },
    },
    lineItems: {
        type: Type.ARRAY, description: "All individual line items from the charges/details section.",
        items: {
            type: Type.OBJECT, properties: {
                description: { type: Type.STRING, description: "The description of the charge or credit." },
                amount: { type: Type.NUMBER, description: "The corresponding amount. Use negative numbers for payments or credits." },
            }, required: ["description", "amount"],
        }
    },
    verificationQuestions: {
        type: Type.ARRAY, description: "If you are uncertain about a specific value due to blurriness or ambiguity, create a question for the user to verify it. Only create questions for critical data points you are unsure about.",
        items: {
            type: Type.OBJECT, properties: {
                field: { type: Type.STRING, description: "A dot-notation path to the uncertain field (e.g., 'usageCharts.0.data.3.value')." },
                question: { type: Type.STRING, description: "A clear, simple question for the user (e.g., 'Is the usage for Sep 2017 approximately 120 kWh? The bar is blurry.')." }
            }, required: ["field", "question"]
        }
    }
  },
  // FIX: Ensure required fields are string literals and optional fields like verificationQuestions are not listed.
  required: ["accountNumber", "totalCurrentCharges", "usageCharts", "lineItems", "confidenceScore", "confidenceReasoning"],
};

const prompt = `You are a world-class OCR system specializing in analyzing utility bills from any provider worldwide. Your mission is to extract detailed information from the provided bill image with extreme accuracy, even if the image quality is poor (blurry, skewed, low light).

**Core Instructions:**
1.  **Analyze Image**: Scrutinize the provided utility bill image.
2.  **Strict JSON Output**: Your entire response MUST be a single, raw JSON object that conforms to the provided schema. Do not include any introductory text, explanations, or markdown formatting like \`\`\`json.
3.  **Chart Data Estimation**: If bar charts or graphs are present, meticulously estimate the values from the bar heights relative to the y-axis, even if exact numbers aren't printed on the bars.
4.  **Confidence Assessment**: Provide a \`confidenceScore\` (0.0-1.0) and a detailed \`confidenceReasoning\`. Be explicit about what parts of the image (e.g., "the line items section", "the top-right corner with the date") were difficult to read and why (e.g., "due to a camera flash glare", "text is pixelated").
5.  **User Verification**: If you are uncertain about any specific, critical data point (like a charge amount or a usage bar), generate a clear \`verificationQuestions\` item for it. This is your chance to ask the user for help. For instance, if a bar in a chart is blurry, ask "Is the usage for [Month Year] approximately [Your Best Guess]? The bar is hard to read."
6.  **Completeness**: Ensure every required field in the schema is present. If an optional field (like \`accountName\`) is not found, omit it from the final JSON.`;

type BillDataSansId = Omit<BillData, 'id' | 'analyzedAt'>;
interface AnalysisResult {
    parsedData: BillDataSansId;
    rawResponse: string;
}

// FIX: Add helper function to validate and construct Ollama URLs, providing clearer errors.
const getValidatedOllamaUrl = (baseUrl: string, path: string): string => {
    try {
        // Ensure the base URL has a protocol, which is a common user error.
        if (!/^(https?|ftp):\/\//i.test(baseUrl)) {
             throw new Error("URL is missing a protocol (e.g., http:// or https://).");
        }
        const url = new URL(path, baseUrl);
        return url.toString();
    } catch (error) {
        console.error("Invalid Ollama URL provided:", error);
        throw new Error(`The Ollama URL "${baseUrl}" is invalid. Please check the format and try again.`);
    }
};

const postProcessData = (parsedData: any): BillDataSansId => {
    if (typeof parsedData.totalCurrentCharges === 'string') {
        parsedData.totalCurrentCharges = parseFloat(parsedData.totalCurrentCharges.replace(/[^0-9.-]+/g,""));
    }
    // Ensure nested numeric data is also parsed correctly
    if (parsedData.lineItems) {
        parsedData.lineItems.forEach((item: any) => {
            if(typeof item.amount === 'string') {
                item.amount = parseFloat(item.amount.replace(/[^0-9.-]+/g, ""));
            }
        });
    }
    if (parsedData.usageCharts) {
        parsedData.usageCharts.forEach((chart: any) => {
            if (chart.data) {
                chart.data.forEach((point: any) => {
                    if(point.usage) {
                        point.usage.forEach((u: any) => {
                            if(typeof u.value === 'string') {
                                u.value = parseFloat(u.value);
                            }
                        })
                    }
                })
            }
        })
    }
    return parsedData;
};

// --- Gemini Provider ---

const callGemini = async (imageB64: string, addLog: Function): Promise<AnalysisResult> => {
    addLog('INFO', 'Starting bill analysis with Gemini...');
    
    const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
    
    const imagePart = {
        inlineData: {
            mimeType: imageB64.substring(imageB64.indexOf(":") + 1, imageB64.indexOf(";")),
            data: imageB64.substring(imageB64.indexOf(",") + 1),
        },
    };

    try {
        const requestPayload = {
            model: "gemini-2.5-flash",
            contents: { parts: [{ text: prompt }, imagePart] },
            config: {
                responseMimeType: "application/json",
                responseSchema: billSchema,
            },
        };
        addLog('DEBUG', 'Gemini Request Payload:', requestPayload);

        const response = await ai.models.generateContent(requestPayload);
        
        const jsonText = response.text.trim();
        addLog('DEBUG', 'Gemini Raw Response:', jsonText);

        const parsedJson = JSON.parse(jsonText);
        addLog('INFO', 'Successfully parsed Gemini response.', parsedJson);
        const parsedData = postProcessData(parsedJson);
        return { parsedData, rawResponse: jsonText };
    } catch (error) {
        addLog('ERROR', 'Gemini API Error:', error);
        console.error("Gemini API Error:", error);
        throw new Error("Failed to analyze bill with Gemini. The model could not process the image. Check your API key and try a clearer image.");
    }
};

// --- Ollama Provider ---

const callOllama = async (imageB64: string, url: string, model: string, addLog: Function): Promise<AnalysisResult> => {
    if (!url || !model) {
        addLog('ERROR', 'Ollama URL or model is not configured.');
        throw new Error("Ollama URL or model is not configured. Please add it in the settings.");
    }
    addLog('INFO', `Starting bill analysis with Ollama model: ${model}`);

    const systemPrompt = `You are an API that exclusively returns JSON. Do not include any conversational text, explanations, or markdown formatting like \`\`\`json. Your entire response must be a single, raw JSON object that strictly adheres to the provided schema.

User Request: ${prompt}
JSON Schema: ${JSON.stringify(billSchema)}`;

    let endpoint: string;
    try {
        // FIX: Use URL validator for a better error message on malformed URLs.
        endpoint = getValidatedOllamaUrl(url, "/v1/chat/completions");
    } catch (error) {
        addLog('ERROR', 'Invalid Ollama URL in settings', { url, error });
        if (error instanceof Error) throw error;
        throw new Error("An unknown error occurred during Ollama URL validation.");
    }

    try {
        const requestBody = {
            model: model,
            response_format: { type: "json_object" },
            messages: [
                {
                    role: "system",
                    content: systemPrompt,
                },
                {
                    role: "user",
                    content: [
                        { type: "text", text: "Analyze this utility bill image." },
                        { type: "image_url", image_url: { url: imageB64 } }
                    ]
                }
            ],
        };
        addLog('DEBUG', `Ollama Request to ${endpoint}`, requestBody);


        const response = await fetch(endpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestBody),
        });

        if (!response.ok) {
            const errorBody = await response.text();
            addLog('ERROR', `Ollama API Error (${response.status})`, errorBody);
            console.error("Ollama API Error Response:", errorBody);
            throw new Error(`Ollama API returned an error: ${response.status} ${response.statusText}. Check your server URL and ensure the model is running.`);
        }

        const responseData = await response.json();
        addLog('DEBUG', 'Ollama Raw Response:', responseData);
        const jsonText = responseData.choices[0].message.content;
        const parsedJson = JSON.parse(jsonText);
        addLog('INFO', 'Successfully parsed Ollama response.', parsedJson);
        const parsedData = postProcessData(parsedJson);
        return { parsedData, rawResponse: jsonText };

    } catch (error) {
        addLog('ERROR', 'Ollama Request Error:', error);
        console.error("Ollama Request Error:", error);
        if (error instanceof TypeError) {
             // FIX: Refine error message to be more specific about CORS.
             throw new Error("Could not connect to the Ollama server. This is often a network or CORS issue. Please ensure: 1) The server is running. 2) The URL is correct. 3) CORS is enabled on the Ollama server (e.g., set OLLAMA_ORIGINS='*').");
        }
        if (error instanceof SyntaxError) {
            throw new Error("Ollama returned invalid JSON. The model may not have followed instructions. Check the debug log.");
        }
        // Re-throw custom errors (like from URL validation) or other unexpected errors.
        if (error instanceof Error) throw error;
        throw new Error("An unknown error occurred while communicating with Ollama.");
    }
};

export const fetchOllamaModels = async (url: string, addLog: Function): Promise<OllamaModel[]> => {
    if (!url) {
        throw new Error("Ollama URL is not provided.");
    }
    addLog('INFO', `Fetching models from Ollama at ${url}`);
    
    let endpoint: string;
    try {
        // FIX: Use URL validator for a better error message on malformed URLs.
        endpoint = getValidatedOllamaUrl(url, "/api/tags");
    } catch (error) {
        addLog('ERROR', 'Invalid Ollama URL in settings', { url, error });
        if (error instanceof Error) throw error;
        throw new Error("An unknown error occurred during Ollama URL validation.");
    }

    try {
        const response = await fetch(endpoint);
        if (!response.ok) {
            const errorBody = await response.text();
            addLog('ERROR', `Failed to fetch Ollama models (${response.status})`, errorBody);
            throw new Error(`Failed to fetch models: ${response.statusText}. The server responded with status ${response.status}.`);
        }
        const data = await response.json();
        addLog('DEBUG', 'Ollama models fetched successfully.', data.models);
        return data.models;
    } catch (error) {
        addLog('ERROR', 'Error fetching Ollama models:', error);
        console.error("Error fetching Ollama models:", error);
        if (error instanceof TypeError) { // "Failed to fetch" is a TypeError
             // FIX: Refine error message to be more specific about CORS.
            throw new Error("Could not connect to the Ollama server. This is often a network or CORS issue. Please ensure: 1) The server is running. 2) The URL is correct. 3) CORS is enabled on the Ollama server (e.g., set OLLAMA_ORIGINS='*').");
        }
        // Re-throw other errors (like from !response.ok or URL validation)
        if (error instanceof Error) {
            throw error;
        }
        throw new Error("An unexpected error occurred while fetching Ollama models. Check the debug log.");
    }
};


// --- Main Service Function ---

export const analyzeBill = async (imageB64: string, settings: AiSettings, addLog: Function): Promise<AnalysisResult> => {
    switch (settings.provider) {
        case 'gemini':
            return callGemini(imageB64, addLog);
        case 'ollama':
            return callOllama(imageB64, settings.ollamaUrl, settings.ollamaModel, addLog);
        default:
            const exhaustiveCheck: never = settings.provider;
            throw new Error(`Invalid AI provider: ${exhaustiveCheck}`);
    }
};
