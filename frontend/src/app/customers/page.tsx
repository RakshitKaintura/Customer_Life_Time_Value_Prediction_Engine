import { Topbar } from "@/components/nav/topbar";
import { CustomerSearch } from "./customer-search";
import { ApiStatusBadge } from "./api-status-badge";

export default function CustomersPage() {
  return (
    <div className="page-container">
      <Topbar
        title="Customer Lookup"
        subtitle="Search any customer for LTV prediction, purchase history, and lookalikes"
        actions={<ApiStatusBadge />}
      />
      <div className="page-content">
        <CustomerSearch />
      </div>
    </div>
  );
}
