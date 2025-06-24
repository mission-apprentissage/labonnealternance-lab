import "./globals.css";
import "react-notion-x/src/styles.css";

// eslint-disable-next-line import/no-named-as-default
import MuiDsfrThemeProvider from "@codegouvfr/react-dsfr/mui";
import { DsfrHead } from "@codegouvfr/react-dsfr/next-appdir/DsfrHead";
import { DsfrProvider } from "@codegouvfr/react-dsfr/next-appdir/DsfrProvider";
import { getHtmlAttributes } from "@codegouvfr/react-dsfr/next-appdir/getHtmlAttributes";
import { captureException } from "@sentry/nextjs";
import type { Metadata, Viewport } from "next";
import Link from "next/link";
import type { PropsWithChildren } from "react";
import { Suspense } from "react";
import type { ISessionJson } from "shared/src/routes/_private/auth.routes";

import { Header } from "@/components/header/Header";
import { publicConfig } from "@/config.public";
import { AuthContextProvider } from "@/context/AuthContext";
import { defaultColorScheme } from "@/theme/defaultColorScheme";
import type { ApiError } from "@/utils/api.utils";
import { apiGet } from "@/utils/api.utils";

import { StartDsfr } from "./StartDsfr";

async function getSession(): Promise<ISessionJson | null> {
  try {
    const session = await apiGet(`/_private/auth/session`, {}, { cache: "no-store" });
    return session;
  } catch (error) {
    if ((error as ApiError).context?.statusCode !== 401) {
      captureException(error);
    }

    return null;
  }
}

export const metadata: Metadata = {
  icons: {
    icon: [{ url: "/favicon.ico" }, { url: "/favicon.svg" }],
    apple: [{ url: "/apple-touch-icon.png" }],
  },
  title: publicConfig.productMeta.productName,
  description: "Un service de la Mission Apprentissage",
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
};

export default async function RootLayout({ children }: PropsWithChildren) {
  const session = await getSession();
  const lang = "fr";
  return (
    <html {...getHtmlAttributes({ defaultColorScheme, lang })}>
      <head>
        <StartDsfr />
        <DsfrHead
          Link={Link}
          preloadFonts={[
            //"Marianne-Light",
            //"Marianne-Light_Italic",
            "Marianne-Regular",
            //"Marianne-Regular_Italic",
            "Marianne-Medium",
            //"Marianne-Medium_Italic",
            "Marianne-Bold",
            //"Marianne-Bold_Italic",
            //"Spectral-Regular",
            //"Spectral-ExtraBold"
          ]}
        />
      </head>
      <body>
        <AuthContextProvider initialSession={session ?? null}>
          <Suspense>
            <DsfrProvider lang={lang}>
              <MuiDsfrThemeProvider>
                <Header />
                {children}
              </MuiDsfrThemeProvider>
            </DsfrProvider>
          </Suspense>
        </AuthContextProvider>
      </body>
    </html>
  );
}
