"use client";
import { Box, Typography } from "@mui/material";
import { ProcessorStatusCronComponent } from "job-processor/dist/react";
import { use } from "react";

import { ProcessorStatusProvider } from "@/app/admin/processeur/components/ProcessorStatusProvider";
import Breadcrumb from "@/components/breadcrumb/Breadcrumb";
import { publicConfig } from "@/config.public";
import { PAGES } from "@/utils/routes.utils";

export default function JobTypePage({ params }: { params: Promise<{ name: string }> }) {
  const { name: rawName } = use(params);
  const name = decodeURIComponent(rawName);
  return (
    <Box>
      <Breadcrumb pages={[PAGES.static.adminProcessor, PAGES.dynamic.adminProcessorCron(name)]} />
      <Typography variant="h2" gutterBottom>
        {PAGES.dynamic.adminProcessorCron(name).getTitle()}
      </Typography>
      <ProcessorStatusProvider>
        {(status) => (
          <ProcessorStatusCronComponent
            name={name}
            status={status}
            baseUrl={new URL(PAGES.static.adminProcessor.getPath(), publicConfig.baseUrl).href}
          />
        )}
      </ProcessorStatusProvider>
    </Box>
  );
}
