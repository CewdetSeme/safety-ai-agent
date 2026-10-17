
import {readFile, writeFile} from "node:fs/promises"

import { GoogleGenAI } from "@google/genai";

import crypto from "node:crypto"

import { sendAlertEmail } from "./email.js";

const ai = new GoogleGenAI({})

const MEMORY_File = "./memory.json"

async function loadMemory(){
    try{
        const data = await readFile(MEMORY_File, "utf8")
        if(!data.trim()){
            return [];
        }
        return JSON.parse(data)
    }
    catch(error){
        if(error.code==="ENOENT"){
            return []                
        }
        throw error;
    }
}

async function saveMemory(){
    await writeFile(MEMORY_File, JSON.stringify(memory,null,2), "utf8")
}

let memory = await loadMemory()

const INCIDENTS_FILE = "./incidents.json"

async function logIncident(incident) {
    let incidents = [];

    try{
        const data = await readFile(INCIDENTS_FILE, "utf8");
        incidents = JSON.parse(data)
        if(! Array.isArray(incidents))
            throw new Error("incidents.json must contain a JSON array")
    }
    catch(error){
        if(error.code === "ENOENT"){
            throw error;
        }
    }

    // avoid duplicate recording incidents
    const duplicate = incidents.some(x => x.message === incident.message)

    if(duplicate){
        console.log("this message already exists")
        return false;
    }


    incidents.push({
        id:crypto.randomUUID(),
        ...incident,
        loggedAt:new Date().toISOString()
    })

    await writeFile(
        INCIDENTS_FILE, JSON.stringify(incidents,null,2), "utf8"
    )
    console.log("Incident saved to incidents.json")
    return true;
}

function observe(input){
    console.log("Agent observed:", input)
    return input
}



async function decide(input){

    const response = await ai.models.generateContent({
        model: "gemini-3.5-flash-lite",
        contents: 
            `
            You are an industrial safety AI agent for a power plant.

            PREVIOUS EVENTS FROM MEMORY:
            ${JSON.stringify(memory, null, 2)}

            NEW MESSAGE:
            "${input}"

            Use previous events as context when analyzing the new message.

            Priority rules:
            - HIGH: fire, gas leak, injury, electrical hazard,
            or immediate danger to people or equipment.
            - NORMAL: routine meetings and non-urgent information.

            Return exactly this format:

            PRIORITY: HIGH or NORMAL
            REASON: short explanation
            ACTION: NOTIFY or IGNORE
            `
         
    })

    return response.text.trim()

}



function parseDecision(text){
    const priority = text.match(/PRIORITY:\s*(HIGH|NORMAL)/i)?.[1]?.toUpperCase()

    const action =
        text.match(/ACTION:\s*(NOTIFY|IGNORE)/i)?.[1]?.toUpperCase()

        const reason =
        text.match(/REASON:\s*(.*)/i)?.[1]?.trim();

        return {
        priority,
        reason,
        action
    };
}

function act(decision){
    if(decision.action=="NOTIFY"){
        console.log("Important message.")
        console.log("Priority: ", decision.priority)
        console.log("Reason: ", decision.reason)
        
    }
        
    else
        console.log("No action required")
}

async function runAgent(input){
    try{
        const observation = observe(input);

        const aiResult = await decide(observation)

        const decision = parseDecision(aiResult)

        memory.push({message:observation, decision:decision})

        await saveMemory()

        console.log("Agent Decision: ", decision)

        console.log("Agent memory: ", memory)

        act(decision);

        if(decision.action==="NOTIFY"){
            const isNewIncident = await logIncident({
                message: observation,
                priority: decision.priority,
                reason: decision.reason
            })

            if(isNewIncident){
                await sendAlertEmail({
                    type:"Industrial Safety Incident",
                    location:"Power Plant",
                    severity:decision.priority,
                    message:observation,
                    reason: decision.reason
                })
                console.log("Email sent for new incident.")
            }
            else{
                console.log("Duplicate incident: email not sent.")
            }
        }

    }
    catch(error){
        console.log(error.message)
    }


}



await runAgent("Fire detected near the gas turbine");

await runAgent("Fire detected near the gas turbine");
