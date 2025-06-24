"use client";
import { Box, Typography } from "@mui/material";
import Link from "next/link";
import { Suspense } from "react";

import { NotFound } from "@/icons/NotFound";

import { PAGES } from "../utils/routes.utils";

export default function NotFoundPage() {
  return (
    <Suspense>
      <Box>
        <Box
          padding={8}
          display="flex"
          justifyContent="center"
          flexDirection="column"
          margin="auto"
          maxWidth="600px"
          textAlign="center"
        >
          <NotFound />

          <Box mt={4}>
            <Typography variant="h1" gutterBottom>
              Page non trouvée
            </Typography>

            <Typography>La page que vous recherchez n'existe pas ou a été déplacée</Typography>

            <Box mt={2}>
              <Link href={PAGES.static.home.getPath()}>Retourner à la page d'accueil</Link>
            </Box>
          </Box>
        </Box>
      </Box>
    </Suspense>
  );
}
