import { z } from "zod";

import type { IRoutesDef } from "./common.routes";

export const zCoreRoutes = {
  get: {
    "/healthcheck": {
      method: "get",
      path: "/healthcheck",
      response: {
        "200": z
          .object({
            name: z.string(),
            version: z.string(),
            env: z.enum(["local", "recette", "production", "preview", "test"]),
          })
          .describe("Statut de l'application")
          .strict(),
      },
      securityScheme: null,
      openapi: {
        tags: ["Système"] as string[],
      },
    },
  },
} as const satisfies IRoutesDef;
