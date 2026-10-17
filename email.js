
import nodemailer from "nodemailer";

export async function sendAlertEmail(incident) {
    const transporter = nodemailer.createTransport({
        service: "gmail",
        auth: {
            user: process.env.GMAIL_USER,
            pass: process.env.GMAIL_APP_PASSWORD
        }
    });

    await transporter.sendMail({
        from: process.env.GMAIL_USER,
        to: process.env.GMAIL_USER,
        subject: `Safety Alert: ${incident.message}`,
        text: `
            Safety incident detected!

            Message: ${incident.message}
            Reason: ${incident.reason}
            Time: ${new Date().toLocaleString()}
                    `
    });

    console.log("Safety alert email sent.");
}