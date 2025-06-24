"use client";

import { Suspense } from "react";

import Breadcrumb from "@/components/breadcrumb/Breadcrumb";

import { PAGES } from "../../utils/routes.utils";
import DonneesPersonnelles from "./components/DonneesPersonnelles";

const DonneesPersonnellesPage = () => {
  return (
    <Suspense>
      <Breadcrumb pages={[PAGES.static.donneesPersonnelles]} />
      <DonneesPersonnelles />
    </Suspense>
  );
};
export default DonneesPersonnellesPage;
