"use client";

import { Suspense } from "react";

import Breadcrumb from "@/components/breadcrumb/Breadcrumb";

import { PAGES } from "../../utils/routes.utils";
import Cgu from "./components/Cgu";

const CGUPage = () => {
  return (
    <Suspense>
      <Breadcrumb pages={[PAGES.static.cgu]} />
      <Cgu />
    </Suspense>
  );
};
export default CGUPage;
