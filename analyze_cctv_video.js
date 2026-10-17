import { GoogleGenAI } from "@google/genai";

import {execFile} from "node:child_process"

import { promisify } from "node:util";

import {mkdir,readFile, readdir, rm, writeFile} from "node:fs/promises"

import path from "node:path"

// Convert execFile's callback API into an awaitable function.
// We do not need to write new Promise(), resolve, or reject.
const execFileAsync = promisify(execFile);

// Create the Gemini AI client.
// It reads your configured API key from the environment.
const ai = new GoogleGenAI({});

const VIDEO_FILE = "./cctv-test.mp4";

// Extract one image every 2 seconds.
const INTERVAL_SECONDS = 3;

// Folder where FFmpeg will save the images.
const FRAMES_DIR = "./video-frames";

// File where the final results will be saved.
const REPORT_FILE = "./cctv-safety-report.json";

// Use the same working Gemini model as your successful
// single-image analysis.
const MODEL_NAME = "gemini-3.5-flash-lite";



async function extractFrames(){

    // Remove old frames so that previous video images
    // cannot accidentally appear in the new report.
    await rm(FRAMES_DIR, {recursive:true, force:true})

    // Create a fresh folder for this video's images.
    await mkdir(FRAMES_DIR, {recursive: true})

    console.log("Extracting CCTV frames...");

    // Run FFmpeg and wait for it to finish.
    // No explicit Promise, resolve, or reject is needed.
    await execFileAsync( "ffmpeg", [
        "-hide_banner",
        "-loglevel", "error",
        //inout video
        "-i", VIDEO_FILE,
        // Sample one frame every INTERVAL_SECONDS.
        "-vf", `fps=1/${INTERVAL_SECONDS}`,
        // High-quality JPEG output.
        "-q:v", "2",
        // Output names: frame_0001.jpg, frame_0002.jpg, ...
        path.join(FRAMES_DIR, "frame_%04d.jpg")

    ] )

    // Read the folder and keep only our generated JPEG frames.
    const files = (await readdir(FRAMES_DIR))
        .filter(file => /^frame_\d+\.jpg$/i.test(file))
        .sort();

        // Stop if no images were created.
    if (files.length === 0) {
        throw new Error(
            "No frames extracted. Check the MP4 file."
        );
    }
    console.log(`Extracted ${files.length} frames.`);

    // Return filenames for the next step.
    return files;

}// end function


// function to analyze a frame
async function analyzeFrame(filename, index){

    // Read the JPEG image from the frames folder.
    const imagePath = path.join(FRAMES_DIR, filename);
    const image = await readFile(imagePath);

    // Convert the image bytes to Base64 for Gemini.
    const imageBase64 = image.toString("base64");

    console.log(`Analyzing ${filename}...`);

    // Send both the image and our safety instructions.
    const response = await ai.models.generateContent({
        model: MODEL_NAME,

        contents: [
            {
                inlineData: {
                    mimeType: "image/jpeg",
                    data: imageBase64
                }
            },
            {
                text: `
                You are an industrial safety observer at a gas power plant.

Analyze this CCTV image.

Report:
1. Number of visible people.
2. Hard hats: COMPLIANT, MISSING, or UNCERTAIN.
3. High-visibility vests: COMPLIANT, MISSING, or UNCERTAIN.
4. Other clearly visible safety hazards.

Rules:
- Report MISSING only when the absence is clearly visible.
- If a person's head or body is obscured, report UNCERTAIN.
- Do not invent hazards that cannot be seen.
- Separate confirmed observations from uncertainty.
- Be concise and factual.

Return a structured safety report.
                `
            }
              ]
    });

    // Return the result along with the frame's approximate time.
    return {
        frame: filename,
        timeSeconds: index * INTERVAL_SECONDS,
        report: response.text ?? "No report returned."
    };

}// end function



async function main() {

    try {
        // First, extract all images from the video.
        const frames = await extractFrames();

        // Store every frame's result here.
        const reports = [];

        // Analyze frames one at a time.
        // Sequential processing is simpler and avoids
        // sending too many Gemini requests simultaneously.
        for (let i = 0; i < frames.length; i++) {

            try {
                const result = await analyzeFrame(frames[i], i);

                // Save this frame's analysis in memory.
                reports.push(result);

                // Display the result in the terminal.
                console.log(result);

            } catch (error) {

                // A failed frame should not stop the
                // analysis of the remaining frames.
                console.error(
                    `Failed to analyze ${frames[i]}:`,
                    error.message
                );

                reports.push({
                    frame: frames[i],
                    timeSeconds: i * INTERVAL_SECONDS,
                    error: error.message
                });
            }
        }

        // Save all frame results to a JSON file.
        await writeFile(
            REPORT_FILE,
            JSON.stringify(reports, null, 2),
            "utf8"
        );

        console.log(
            `Analysis complete. Report saved to ${REPORT_FILE}`
        );

    } catch (error) {

        // Handles major failures such as missing video files
        // or FFmpeg extraction errors.
        console.error(
            "Video analysis failed:",
            error.message
        );
    }
}

// Start the program.
await main();