import { addJob, initJobProcessor } from "job-processor";

import config from "@/config";
import logger, { createJobProcessorLogger } from "@/services/logger";
import { createIndexes, getDatabase } from "@/services/mongodb/mongodbService";

import { validateModels } from "./db/schemaValidation";
import { create as createMigration, status as statusMigration, up as upMigration } from "./migrations/migrations";

export async function setupJobProcessor() {
  return initJobProcessor({
    db: getDatabase(),
    logger: createJobProcessorLogger(logger),
    crons: config.env === "preview" || config.env === "local" ? {} : {},
    jobs: {
      "indexes:recreate": {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        handler: async (job) => createIndexes(job.payload as any),
      },
      "db:validate": {
        handler: async () => validateModels(),
      },
      "migrations:up": {
        handler: async () => {
          await upMigration();
          // Validate all documents after the migration
          await addJob({ name: "db:validate", queued: true });
          return;
        },
      },
      "migrations:status": {
        handler: async () => {
          const pendingMigrations = await statusMigration();
          console.log(`migrations-status=${pendingMigrations === 0 ? "synced" : "pending"}`);
          return;
        },
      },
      "migrations:create": {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        handler: async (job) => createMigration(job.payload as any),
      },
    },
  });
}
