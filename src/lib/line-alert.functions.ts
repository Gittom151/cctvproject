import { createServerFn } from "@tanstack/react-start";
import { z } from "zod";

const payloadSchema = z.object({
  camera: z.number().int().min(1).max(64),
  reason: z.string().min(1).max(300),
  time: z.string().min(1).max(64),
  latitude: z.number().min(-90).max(90),
  longitude: z.number().min(-180).max(180),
  locationName: z.string().min(1).max(100),
  snapshot: z.string().max(4_000_000).optional(),
});

const LINE_BROADCAST_URL = "https://api.line.me/v2/bot/message/broadcast";
const SNAPSHOT_BUCKET = "incident-snapshots";

function decodeBase64Image(dataUrl: string): Uint8Array | null {
  const base64 = dataUrl.includes(",") ? dataUrl.slice(dataUrl.indexOf(",") + 1) : dataUrl;
  try {
    const binary = atob(base64);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
    return bytes.length > 0 ? bytes : null;
  } catch {
    return null;
  }
}

export const sendLineAccidentAlert = createServerFn({ method: "POST" })
  .inputValidator((data: unknown) => payloadSchema.parse(data))
  .handler(async ({ data }) => {
    const token = process.env["LINE_CHANNEL_ACCESS_TOKEN"];
    if (!token) {
      return { ok: false as const, error: "ยังไม่ได้ตั้งค่า LINE Channel Access Token" };
    }

    let imageUrl: string | null = null;
    if (data.snapshot) {
      const bytes = decodeBase64Image(data.snapshot);
      if (bytes) {
        try {
          const { supabaseAdmin } = await import("@/integrations/supabase/client.server");
          const path = `cam-${data.camera}/${Date.now()}.jpg`;
          const upload = await supabaseAdmin.storage
            .from(SNAPSHOT_BUCKET)
            .upload(path, bytes, { contentType: "image/jpeg", upsert: true });
          if (!upload.error) {
            const signed = await supabaseAdmin.storage
              .from(SNAPSHOT_BUCKET)
              .createSignedUrl(path, 60 * 60 * 24 * 7);
            imageUrl = signed.data?.signedUrl ?? null;
          } else {
            console.error("snapshot upload failed", upload.error);
          }
        } catch (error) {
          console.error("snapshot handling failed", error);
        }
      }
    }

    const mapUrl = `https://www.google.com/maps?q=${data.latitude},${data.longitude}`;
    const messages: unknown[] = [
      {
        type: "text",
        text: [
          "🚨 ยืนยันอุบัติเหตุจากกล้อง CCTV",
          `กล้อง: CAM-0${data.camera}`,
          `ลักษณะเหตุ: ${data.reason}`,
          `เวลา: ${data.time}`,
          `จุดเกิดเหตุ: ${data.locationName}`,
          `พิกัด: ${data.latitude.toFixed(6)}, ${data.longitude.toFixed(6)}`,
          mapUrl,
        ].join("\n"),
      },
      {
        type: "location",
        title: `จุดเกิดเหตุ CAM-0${data.camera}`,
        address: data.locationName,
        latitude: data.latitude,
        longitude: data.longitude,
      },
    ];

    if (imageUrl) {
      messages.push({
        type: "image",
        originalContentUrl: imageUrl,
        previewImageUrl: imageUrl,
      });
    }

    try {
      const response = await fetch(LINE_BROADCAST_URL, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({ messages }),
      });

      if (!response.ok) {
        const detail = await response.text();
        console.error("LINE broadcast failed", response.status, detail);
        return {
          ok: false as const,
          error: `ส่งเข้าไลน์ไม่สำเร็จ (สถานะ ${response.status})`,
        };
      }

      return { ok: true as const, imageSent: Boolean(imageUrl) };
    } catch (error) {
      console.error("LINE broadcast error", error);
      return { ok: false as const, error: "ไม่สามารถเชื่อมต่อระบบไลน์ได้" };
    }
  });
