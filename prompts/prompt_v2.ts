
import { Type } from "@google/genai";

// This is the V2 prompt, designed to be simpler and more robust for a wider range of models.

export const billSchema = {
  type: Type.OBJECT,
  properties: {
    accountName: { type: Type.STRING, description: "Account holder's full name." },
    accountNumber: { type: Type.STRING, description: "The account number." },
    serviceAddress: { type: Type.STRING, description: "The full service address." },
    statementDate: { type: Type.STRING, description: "The main date of the bill statement (e.g., 'October 5, 2017')." },
    servicePeriodStart: { type: Type.STRING, description: "The start date of the service period (e.g., 'MM/DD/YYYY')." },
    servicePeriodEnd: { type: Type.STRING, description: "The end date of the service period (e.g., 'MM/DD/YYYY')." },
    totalCurrentCharges: { type: Type.NUMBER, description: "The total amount due for the current period." },
    dueDate: { type: Type.STRING, description: "The payment due date." },
    confidenceScore: { type: Type.NUMBER, description: "A score from 0.0 to 1.0 representing your confidence in the extracted data's accuracy. 1.0 is highest confidence." },
    confidenceReasoning: { type: Type.STRING, description: "A detailed explanation for the confidence score. Mention specific issues like blurriness, glare, or unusual formatting." },
    usageCharts: {
      type: Type.ARRAY, description: "An array of ALL usage charts found on the bill (e.g., 'Electricity Usage', 'Water Use').",
      items: {
        type: Type.OBJECT, properties: {
          title: { type: Type.STRING, description: "The title of the chart." },
          unit: { type: Type.STRING, description: "The unit of measurement (e.g., kWh, m³)." },
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
        type: Type.ARRAY, description: "If you are uncertain about a specific value due to blurriness, create a question for the user to verify it.",
        items: {
            type: Type.OBJECT, properties: {
                field: { type: Type.STRING, description: "A dot-notation path to the uncertain field (e.g., 'usageCharts.0.data.3.usage.0.value')." },
                question: { type: Type.STRING, description: "A clear, simple question for the user (e.g., 'Is the usage for Sep 2017 approximately 120 kWh? The bar is blurry.')." }
            }, required: ["field", "question"]
        }
    }
  },
  required: ["accountNumber", "totalCurrentCharges", "usageCharts", "lineItems", "confidenceScore", "confidenceReasoning"],
};

export const prompt = `You are an expert OCR system specializing in utility bills. Your task is to extract information from the provided image and respond ONLY with a single, raw JSON object that conforms to the schema. Do not include any other text, explanations, or markdown.

**Extraction Steps:**
1.  **Extract Key Details**: Find the account number, total current charges, statement date, and other primary details.
2.  **Find ALL Usage Charts**: The bill may have multiple charts (e.g., for electricity and water). You MUST find and extract data for all of them.
3.  **Extract Chart Data**: For each chart, extract the title, unit, and all monthly data points. Bar charts often show usage for multiple years for the same month. You MUST extract the data for each year as a nested object.
    - **Correct nested 'usage' example**: \`"usage": [{ "year": "2016", "value": 1400 }, { "year": "2017", "value": 1350 }]\`
4.  **Extract Line Items**: List all individual charges and credits from the bill summary.
5.  **Assess Confidence**: Provide a \`confidenceScore\` (0.0-1.0) and a detailed \`confidenceReasoning\`, mentioning any specific parts of the bill that were hard to read.
6.  **Create Verification Questions**: If you are uncertain about any specific value, create an item in the \`verificationQuestions\` array with the field path and a clear question for the user.

Your final output must be nothing but the JSON object.`;
