import type { MetadataRoute } from "next";

import { publicConfig } from "@/config.public";

export interface IPage {
  getPath: () => string;
  index: boolean;
  getTitle: () => string;
}

export interface INotionPage extends IPage {
  notionId: string;
}

export interface IPages {
  static: Record<string, IPage>;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  dynamic: Record<string, (params: any) => IPage>;
  notion: Record<string, INotionPage>;
}

export const PAGES = {
  static: {
    home: {
      getPath: () => `/` as string,
      index: true,
      getTitle: () => "Accueil",
    },
    explorerApi: {
      getPath: () => `/explorer` as string,
      index: true,
      getTitle: () => "Explorer",
    },
    mentionsLegales: {
      getPath: () => `/mentions-legales` as string,
      index: true,
      getTitle: () => "Mentions Légales",
    },
    accessibilite: {
      getPath: () => `/accessibilite` as string,
      index: true,
      getTitle: () => "Accessibilité",
    },
    cgu: {
      getPath: () => `/cgu` as string,
      index: true,
      getTitle: () => "Conditions Générales d'Utilisation",
    },
    donneesPersonnelles: {
      getPath: () => `/donnees-personnelles` as string,
      index: true,
      getTitle: () => "Données Personnelles",
    },
    politiqueConfidentialite: {
      getPath: () => `/politique-confidentialite` as string,
      index: true,
      getTitle: () => "Politique de Confidentialité",
    },
    compteProfil: {
      getPath: () => `/compte/profil` as string,
      index: true,
      getTitle: () => "Mon profil",
    },
    connexion: {
      getPath: () => `/auth/connexion` as string,
      index: true,
      getTitle: () => "Connexion",
    },
    motDePasseOublie: {
      getPath: () => `/auth/mot-de-passe-oublie` as string,
      index: true,
      getTitle: () => "Mot de passe oublié",
    },
    adminUsers: {
      getPath: () => `/admin/utilisateurs` as string,
      index: false,
      getTitle: () => "Gestion des utilisateurs",
    },
    adminOrganisations: {
      getPath: () => `/admin/organisations` as string,
      index: false,
      getTitle: () => "Gestion des organisations",
    },
    adminProcessor: {
      getPath: () => `/admin/processeur` as string,
      index: false,
      getTitle: () => "Administration du processeur",
    },
  },
  dynamic: {
    inscription: (token: string): IPage => ({
      getPath: () => `/auth/inscription?token=${token}`,
      index: false,
      getTitle: () => "Inscription",
    }),
    refusInscription: (token: string): IPage => ({
      getPath: () => `/auth/refus-inscription?token=${token}`,
      index: false,
      getTitle: () => "Inscription",
    }),
    adminUserView: (id: string): IPage => ({
      getPath: () => `/admin/utilisateurs/${id}`,
      index: false,
      getTitle: () => "Fiche utilisateur",
    }),
    adminOrganisationView: (id: string): IPage => ({
      getPath: () => `/admin/organisations/${id}`,
      index: false,
      getTitle: () => "Fiche Organisation",
    }),
    adminProcessorJob: (name: string): IPage => ({
      getPath: () => `/admin/processeur/job/${name}`,
      index: false,
      getTitle: () => `Job ${name}`,
    }),
    adminProcessorJobInstance: (params: { name: string; id: string }): IPage => ({
      getPath: () => `/admin/processeur/job/${params.name}/${params.id}`,
      index: false,
      getTitle: () => `Tâche Job ${params.id}`,
    }),
    adminProcessorCron: (name: string): IPage => ({
      getPath: () => `/admin/processeur/cron/${name}`,
      index: false,
      getTitle: () => `CRON ${name}`,
    }),
    adminProcessorCronTask: (params: { name: string; id: string }): IPage => ({
      getPath: () => `/admin/processeur/cron/${params.name}/${params.id}`,
      index: false,
      getTitle: () => `Tâche CRON ${params.id}`,
    }),
  },
  notion: {},
} as const satisfies IPages;

function getRawPath(pathname: string): string {
  const rawPath = pathname.replace(/^\/fr/, "").replace(/^\/en/, "");
  return rawPath === "" ? "/" : rawPath;
}

export function isStaticPage(pathname: string): boolean {
  return Object.values(PAGES.static).some((page) => getRawPath(page.getPath()) === pathname);
}

export function isDynamicPage(pathname: string): boolean {
  if (pathname === "/auth/inscription") {
    return true;
  }
  if (pathname === "/auth/refus-inscription") {
    return true;
  }
  if (/^\/admin\/utilisateurs\/[^/]+$/.test(pathname)) {
    return true;
  }
  if (/^\/admin\/organisations\/[^/]+$/.test(pathname)) {
    return true;
  }
  if (/^\/admin\/processeur\/job\/[^/]+$/.test(pathname)) {
    return true;
  }
  if (/^\/admin\/processeur\/job\/[^/]+\/[^/]+$/.test(pathname)) {
    return true;
  }
  if (/^\/admin\/processeur\/cron\/[^/]+$/.test(pathname)) {
    return true;
  }
  if (/^\/admin\/processeur\/cron\/[^/]+\/[^/]+$/.test(pathname)) {
    return true;
  }
  if (/^\/admin\/importers\/[^/]+$/.test(pathname)) {
    return true;
  }

  return false;
}

export function isNotionPage(pathname: string): boolean {
  return pathname.startsWith("/doc/") || /^\/notion\/[^/]+$/.test(pathname);
}

function getSitemapItem(page: IPage): MetadataRoute.Sitemap[number] {
  return {
    url: `${publicConfig.baseUrl}${getRawPath(page.getPath())}`,
    alternates: {
      languages: {
        fr: `${publicConfig.baseUrl}${page.getPath()}`,
      },
    },
  };
}

export function getSitemap(): MetadataRoute.Sitemap {
  return Object.values(PAGES.static)
    .filter((page) => page.index)
    .map(getSitemapItem);
}

export function isPage(pathname: string): boolean {
  return isStaticPage(pathname) || isDynamicPage(pathname) || isNotionPage(pathname);
}
