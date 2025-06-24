import type { Server } from "@/server/server";

export function apiKeyUsageMiddleware(server: Server) {
  server.addHook("onSend", async (request, _reply) => {
    if (request.user && request.user.type === "user" && request.api_key) {
      // We group usage by day
      const now = new Date();
      now.setUTCMilliseconds(0);
      now.setUTCSeconds(0);
      now.setUTCMinutes(0);
      now.setUTCHours(0);
    }
  });
}
