"use client";

import { Suspense } from "react";

import Breadcrumb from "@/components/breadcrumb/Breadcrumb";

import { PAGES } from "../../utils/routes.utils";
import Explorer from "./components/Explorer";

const ExplorerPage = () => {
  return (
    <Suspense>
      <Breadcrumb pages={[PAGES.static.mentionsLegales]} />
      <Explorer />
    </Suspense>
  );
};
export default ExplorerPage;
