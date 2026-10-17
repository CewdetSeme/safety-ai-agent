// function to read image and send it to AI to check any unsafe act
import {readFile} from "node:fs/promises"

import { GoogleGenAI } from "@google/genai";
const ai = new GoogleGenAI({})

async function analyzeCCTV(imagePath){
    const image = await readFile(imagePath)

    const image64Base = image.toString("base64")

    const response = await ai.models.generateContent({
        model: "gemini-3.5-flash-lite",
        contents: [
            {
                inlineData:{
                    mimeType: "image/jpeg",
                    data: image64Base
                }
            },
            {
                text: `
                    You are an industrial safety observer at a power plant.

                    Inspect this CCTV image and report:

                    1. How many people are visible?
                    2. Is each person's hard hat visible?
                    3. Is each person wearing a high-visibility vest?
                    4. Are any visible unsafe conditions present?

                    For each person's PPE, use one of:
                    - COMPLIANT: clearly visible and apparently worn correctly
                    - MISSING: clearly absent
                    - UNCERTAIN: cannot determine from the image

                    Do not guess when a person or item is obscured.
                    Distinguish confirmed hazards from uncertain observations.

                    Return a clear, numbered report.
                `
            }
        ]
    })

    console.log("CCTV Report")
    console.log(response.text)

}// end fucntion

analyzeCCTV("./cctv_image.jpg").catch(error => {console.log(error.message)})

