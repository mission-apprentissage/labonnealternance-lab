"use client";

import { Suspense } from "react";

import Breadcrumb from "@/components/breadcrumb/Breadcrumb";

import { PAGES } from "../../utils/routes.utils";
import MentionsLegales from "./components/MentionLegales";

const MentionsLegalesPage = () => {
  return (
    <Suspense>
      <Breadcrumb pages={[PAGES.static.mentionsLegales]} />
      <MentionsLegales />
    </Suspense>
  );
};
export default MentionsLegalesPage;
