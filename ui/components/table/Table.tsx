"use client";

import { Box } from "@mui/material";
import { styled } from "@mui/material/styles";
import type { DataGridProps, GridValidRowModel } from "@mui/x-data-grid";
import { DataGrid } from "@mui/x-data-grid";
import { frFR } from "@mui/x-data-grid/locales";
import type { ReactNode, RefObject } from "react";

const StyledDataGrid = styled(DataGrid)(({ theme }) => ({
  "& .MuiTablePagination-selectLabel": {
    margin: theme.spacing(0),
  },
}));

export const WrappeDataGridAction = ({
  children,
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  touchRippleRef,
}: {
  children: ReactNode | undefined;
  touchRippleRef?: RefObject<unknown>;
}) => {
  return <>{children}</>;
};

export const Table = <R extends GridValidRowModel>(props: DataGridProps<R>) => {
  return (
    <Box my={2}>
      <StyledDataGrid
        // @ts-expect-error: .
        getRowId={(row) => row?._id ?? row.id}
        rowHeight={60}
        localeText={frFR.components.MuiDataGrid.defaultProps.localeText}
        {...props}
      />
    </Box>
  );
};
